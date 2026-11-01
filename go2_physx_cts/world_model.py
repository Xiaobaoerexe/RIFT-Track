"""Encoded physical dynamics with full pose and supervised contact telemetry.

No task label enters dynamics. Geometry is sampled at each autoregressively
predicted pose; terminal transitions remain supervised, reset jumps do not.
"""
import torch
from torch import nn
from torch.nn import functional as F
from . import state as st
from .networks import RunningNorm, GeometryEncoder, mlp
from .terrain import geometry_crop
from .precision import network_autocast


def supervised_loss(model,window,field,default_q):
    """Compilation boundary for one unchanged autoregressive supervision update.

    Keep the module itself unwrapped so parameter ownership, running statistics
    and checkpoint keys remain identical. Diagnostics may call model.loss with
    return_terms directly without compiling a second full supervision graph.
    """
    return model.loss(window,field,default_q)


class WorldModel(nn.Module):
    contact_dim=14
    def __init__(self,hidden=(1024,1024,1024),mixed_precision=True):
        super().__init__();self.mixed_precision=mixed_precision
        self.physical_norm=RunningNorm(st.STATE_DIM+4)
        self.action_norm=RunningNorm(25)
        self.context_norm=RunningNorm(st.CONTEXT_DIM)
        self.physical_encoder=mlp(st.STATE_DIM+4,(128,),128)
        self.action_encoder=mlp(25,(64,),64)
        self.context_encoder=mlp(st.CONTEXT_DIM,(32,),32)
        self.geometry_encoder=GeometryEncoder(128,mixed_precision=mixed_precision)
        self.output_dim=st.STATE_DIM+st.POSE_DELTA_DIM
        self.net=mlp(352,hidden,self.output_dim+24+self.contact_dim)
        self.out_norm=RunningNorm(self.output_dim)
        self.telemetry_norm=RunningNorm(24)
    def _features(self,s,a,a_prev,xi,local_map,pose):
        delayed=torch.cat((a,a_prev,xi[:,3:4]),-1)
        return torch.cat((self.physical_encoder(self.physical_norm(torch.cat((s,pose[:,3:]),-1))),
            self.action_encoder(self.action_norm(delayed)),self.context_encoder(self.context_norm(xi)),
            self.geometry_encoder(local_map)),-1)
    @torch.no_grad()
    def update_stats(self,s,a,a_prev,xi,local_map,s_next,pose_delta,telemetry=None,pose=None):
        self.physical_norm.update(torch.cat((s,pose[:,3:]),-1)); self.action_norm.update(torch.cat((a,a_prev,xi[:,3:4]),-1)); self.context_norm.update(xi)
        self.out_norm.update(torch.cat((s_next-s,pose_delta),-1))
        if telemetry is not None:
            self.telemetry_norm.update(torch.cat((telemetry['applied_torque'],telemetry['foot_velocity'].reshape(-1,12)),-1))
    def predict(self,s,a,a_prev,xi,local_map,pose):
        with network_autocast(s,self.mixed_precision):y=self.net(self._features(s,a,a_prev,xi,local_map,pose))
        # Autoregressive integration, contact probabilities and all supervised
        # losses operate onFP32 physical quantities, not accumulated BF16 state.
        y=y.float()
        physical=self.out_norm.unnormalize(y[:,:self.output_dim])
        auxiliary=self.telemetry_norm.unnormalize(y[:,self.output_dim:self.output_dim+24])
        logits=y[:,self.output_dim+24:]
        telemetry={'applied_torque':auxiliary[:,:12],'foot_velocity':auxiliary[:,12:].reshape(-1,4,3),
                   'contact_prob':logits.sigmoid(),'contact_logits':logits}
        return s+physical[:,:st.STATE_DIM],physical[:,st.STATE_DIM:],telemetry
    def forward(self,s,a,a_prev,xi,local_map,pose):
        ns,dp,_=self.predict(s,a,a_prev,xi,local_map,pose); return ns,dp
    def loss(self,w,field=None,default_q=None,return_terms=False):
        s=w['s'][:,0]; pose=w['pose'][:,0]; total=s.new_zeros(())
        sums={name:s.new_zeros(()) for name in ('state','pose','contact','telemetry','consistency')}
        scale=self.out_norm.var.sqrt().clamp_min(1e-4)
        telemetry_scale=self.telemetry_norm.var.sqrt().clamp_min(1e-3)
        valid=w['valid']; denom=valid.sum().clamp_min(1)
        for j in range(w['a'].shape[1]):
            xi=w['xi'][:,j] if w['xi'].ndim==3 else w['xi']
            local=geometry_crop(s,pose,field,kind='world_model')
            raw,dp,telemetry=self.predict(s,w['a'][:,j],w['a_prev'][:,j],xi,local,pose)
            next_pose=st.apply_pose_delta(pose,dp)
            ns=st.reconcile_state(raw,next_pose,field,default_q)
            target_s=w['s_next'][:,j]; target_pose=w['pose_next'][:,j]
            state_loss=((ns-target_s)/scale[:st.STATE_DIM]).square().mean(-1)
            # Compare accumulated SE(3) prediction against each measured pose.
            pose_error=st.pose_delta(target_pose,next_pose)
            pose_loss=(pose_error/scale[st.STATE_DIM:]).square().mean(-1)
            consistency=((raw-ns)/scale[:st.STATE_DIM]).square().mean(-1)
            contact=F.binary_cross_entropy_with_logits(telemetry['contact_logits'],w['contact_prob'][:,j].float(),reduction='none').mean(-1)
            auxiliary=torch.cat((telemetry['applied_torque'],telemetry['foot_velocity'].reshape(-1,12)),-1)
            target_aux=torch.cat((w['applied_torque'][:,j],w['foot_velocity'][:,j].reshape(-1,12)),-1)
            aux_loss=((auxiliary-target_aux)/telemetry_scale).square().mean(-1)
            losses={'state':state_loss,'pose':pose_loss,'contact':contact,'telemetry':aux_loss,'consistency':consistency}
            for name,loss in losses.items(): sums[name]=sums[name]+(loss*valid[:,j]).sum()
            total=total+((state_loss+pose_loss+contact+aux_loss+.1*consistency)*valid[:,j]).sum()
            s,pose=ns,next_pose
        result=total/denom
        return (result,{k:v/denom for k,v in sums.items()}) if return_terms else result
