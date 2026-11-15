"""Full-pose physical model plus every-control-step camera latent dynamics.

No task ID enters dynamics. The 1024x3 backbone is retained; only its interfaces
grow. Physical telemetry includes the exact AS2 reward measurements. Visual
targets are re-encoded real captures; a training-only decoder prevents collapse.
"""
import torch
from torch import nn
from torch.nn import functional as F
from . import state as st
from .networks import RunningNorm, GeometryEncoder, mlp
from .terrain import geometry_crop
from .precision import network_autocast
from .latent import LATENT_DIM, LatentDecoder, prepare_latent_targets


def supervised_loss(model,window,field,default_q,*,depth=None,encoder=None,latent_targets=None):
    """Prepare real images outside compilation; compile the tensor loss if needed."""
    return model.loss(window,field,default_q,depth=depth,encoder=encoder,latent_targets=latent_targets)


def telemetry_vector(telemetry):
    return torch.cat((telemetry['applied_torque'],telemetry['foot_velocity'].reshape(-1,12),
                      telemetry['joint_acceleration'],telemetry['log_contact_force']),-1)


def telemetry_target(window,step):
    return torch.cat((window['applied_torque'][:,step],window['foot_velocity'][:,step].reshape(-1,12),
                      window['joint_acceleration'][:,step],window['log_contact_force'][:,step]),-1)


class WorldModel(nn.Module):
    contact_dim=14
    telemetry_dim=44  # torque12, feet velocityXYZ12, qacc12, limb log1p(force)8
    latent_dim=LATENT_DIM
    def __init__(self,hidden=(1024,1024,1024),mixed_precision=True,depth_period=1):
        super().__init__();self.mixed_precision=mixed_precision;self.depth_period=int(depth_period)
        if self.depth_period<1:raise ValueError('depth_period must be positive')
        self.physical_norm=RunningNorm(st.STATE_DIM+4)
        self.action_norm=RunningNorm(25);self.context_norm=RunningNorm(st.CONTEXT_DIM)
        self.physical_encoder=mlp(st.STATE_DIM+4,(128,),128)
        self.action_encoder=mlp(25,(64,),64);self.context_encoder=mlp(st.CONTEXT_DIM,(32,),32)
        self.geometry_encoder=GeometryEncoder(128,mixed_precision=mixed_precision)
        self.output_dim=st.STATE_DIM+st.POSE_DELTA_DIM
        self.net=mlp(352+LATENT_DIM,hidden,
                     self.output_dim+self.telemetry_dim+self.contact_dim+LATENT_DIM)
        # Instinct initializes predicted increments at zero. Apply that source
        # choice only to the newly added latent rows: the first visual forecast
        # is a hold, rather than an arbitrary delta divided by tiny CNN variance.
        # Keep the existing AS2/depth physical-output initialization unchanged.
        latent_start=self.output_dim+self.telemetry_dim+self.contact_dim
        with torch.no_grad():
            self.net[-1].weight[latent_start:].zero_()
            self.net[-1].bias[latent_start:].zero_()
        self.out_norm=RunningNorm(self.output_dim);self.telemetry_norm=RunningNorm(self.telemetry_dim)
        self.latent_decoder=LatentDecoder(LATENT_DIM,mixed_precision)
        self.register_buffer('z_mean',torch.zeros(LATENT_DIM))
        self.register_buffer('z_std',torch.ones(LATENT_DIM))
        self.register_buffer('dz_std',torch.ones(LATENT_DIM))

    def _features(self,s,a,a_prev,xi,local_map,pose,latent):
        delayed=torch.cat((a,a_prev,xi[:,3:4]),-1)
        if latent is None:
            visual=s.new_zeros((len(s),LATENT_DIM))
        else:
            visual=(latent-self.z_mean)/self.z_std
        return torch.cat((self.physical_encoder(self.physical_norm(torch.cat((s,pose[:,3:]),-1))),
            self.action_encoder(self.action_norm(delayed)),self.context_encoder(self.context_norm(xi)),
            self.geometry_encoder(local_map),visual),-1)

    @torch.no_grad()
    def update_stats(self,s,a,a_prev,xi,local_map,s_next,pose_delta,telemetry=None,pose=None):
        if pose is None:raise ValueError('Full current pose is required')
        self.physical_norm.update(torch.cat((s,pose[:,3:]),-1))
        self.action_norm.update(torch.cat((a,a_prev,xi[:,3:4]),-1));self.context_norm.update(xi)
        self.out_norm.update(torch.cat((s_next-s,pose_delta),-1))
        if telemetry is not None:self.telemetry_norm.update(telemetry_vector(telemetry))

    @torch.no_grad()
    def update_latent_stats(self,targets,rate=.01):
        weight=targets.refresh.to(targets.targets.dtype)[...,None]
        count=weight.sum();total=count+len(targets.initial)
        mean=(targets.initial.sum(0)+(targets.targets*weight).sum((0,1)))/total
        second=(targets.initial.square().sum(0)+(targets.targets.square()*weight).sum((0,1)))/total
        self.z_mean.lerp_(mean,rate)
        self.z_std.lerp_((second-mean.square()).clamp_min(1e-6).sqrt(),rate)
        rms=((targets.increments.square()*weight).sum((0,1))/count.clamp_min(1)).sqrt().clamp_min(1e-3)
        self.dz_std.lerp_(torch.where(count>0,rms,self.dz_std),rate)

    def predict(self,s,a,a_prev,xi,local_map,pose,latent=None):
        with network_autocast(s,self.mixed_precision):
            y=self.net(self._features(s,a,a_prev,xi,local_map,pose,latent))
        y=y.float();end=self.output_dim+self.telemetry_dim
        physical=self.out_norm.unnormalize(y[:,:self.output_dim])
        auxiliary=self.telemetry_norm.unnormalize(y[:,self.output_dim:end])
        logits=y[:,end:end+self.contact_dim]
        telemetry={'applied_torque':auxiliary[:,:12], 'foot_velocity':auxiliary[:,12:24].reshape(-1,4,3),
                   'joint_acceleration':auxiliary[:,24:36], 'log_contact_force':auxiliary[:,36:44],
                   'contact_prob':logits.sigmoid(),'contact_logits':logits}
        telemetry['reward_aux']=torch.cat((auxiliary[:,:12],auxiliary[:,24:36],
                                          telemetry['foot_velocity'][...,:2].reshape(-1,8),auxiliary[:,36:44]),-1)
        next_latent=None
        if latent is not None:
            candidate=latent+y[:,end+self.contact_dim:]*self.dz_std
            next_latent=candidate
        return s+physical[:,:st.STATE_DIM],physical[:,st.STATE_DIM:],telemetry,next_latent

    def forward(self,s,a,a_prev,xi,local_map,pose):
        ns,dp,_,_=self.predict(s,a,a_prev,xi,local_map,pose);return ns,dp

    def _targets(self,w,depth,encoder,latent_targets):
        if latent_targets is not None:return latent_targets
        if (depth is None)!=(encoder is None):raise ValueError('depth and encoder must be supplied together')
        if depth is not None:return prepare_latent_targets(w,depth,encoder,self.depth_period)
        return None

    def _rollout_errors(self,w,field,default_q,scales,latent_targets):
        s=w['s'][:,0];pose=w['pose'][:,0]
        scale,aux_scale=scales
        z=None if latent_targets is None else latent_targets.initial
        rows={name:[] for name in ('state','pose','contact','telemetry','consistency','velocity_mse','pose_mse','latent')}
        if z is not None:
            weight=latent_targets.refresh.to(s.dtype)[...,None];count=weight.sum().clamp_min(1)
            mean=(latent_targets.targets*weight).sum((0,1))/count
            second=(latent_targets.targets.square()*weight).sum((0,1))/count
            z_variance=(second-mean.square()).clamp_min(0).mean().clamp_min(1e-6)
        for j in range(w['a'].shape[1]):
            xi=w['xi'][:,j] if w['xi'].ndim==3 else w['xi']
            raw,dp,telemetry,next_z=self.predict(s,w['a'][:,j],w['a_prev'][:,j],xi,
                geometry_crop(s,pose,field,kind='world_model'),pose,z)
            next_pose=st.apply_pose_delta(pose,dp)
            ns=st.reconcile_state(raw,next_pose,field,default_q)
            pe=st.pose_delta(w['pose_next'][:,j],next_pose)
            latent_error=s.new_zeros(len(s))
            if z is not None:
                dz=torch.where(latent_targets.refresh[:,j,None],next_z-latent_targets.targets[:,j],torch.zeros_like(next_z))
                latent_error=dz.square().mean(-1)/z_variance
            terms={'state':((ns-w['s_next'][:,j])/scale[:st.STATE_DIM]).square().mean(-1),
                'pose':(pe/scale[st.STATE_DIM:]).square().mean(-1),
                'contact':F.binary_cross_entropy_with_logits(telemetry['contact_logits'],w['contact_prob'][:,j].float(),reduction='none').mean(-1),
                'telemetry':((telemetry_vector(telemetry)-telemetry_target(w,j))/aux_scale).square().mean(-1),
                'consistency':((raw-ns)/scale[:st.STATE_DIM]).square().mean(-1),
                'velocity_mse':(ns[:,4:10]-w['s_next'][:,j,4:10]).square().mean(-1),
                'pose_mse':pe.square().mean(-1),
                'latent':latent_error}
            for name,value in terms.items():rows[name].append(value)
            if z is not None:z=next_z
            s,pose=ns,next_pose
        return {name:torch.stack(values,1) for name,values in rows.items()}

    @torch.no_grad()
    def validation_errors(self,w,field,default_q,scales,*,depth=None,encoder=None,latent_targets=None):
        """Read-only errors; neither running scales nor encoder weights are updated."""
        target=self._targets(w,depth,encoder,latent_targets)
        errors=self._rollout_errors(w,field,default_q,scales,target)
        valid=w['valid'];den=valid.sum(-1).clamp_min(1)
        rows={name:(value*valid).sum(-1)/den for name,value in errors.items() if name!='latent'}
        rows['one_step']=errors['state'][:,0]+errors['pose'][:,0]
        if target is not None:
            rows['latent']=(errors['latent']*target.refresh).sum(-1)/target.refresh.sum(-1).clamp_min(1)
        rows['total']=rows['state']+rows['pose']+rows['contact']+rows['telemetry']+.1*rows['consistency']
        return rows

    def loss(self,w,field=None,default_q=None,return_terms=False,*,depth=None,encoder=None,latent_targets=None):
        target=self._targets(w,depth,encoder,latent_targets)
        if target is not None:
            if encoder is None:raise ValueError('Shared depth encoder required for reconstruction')
            self.update_latent_stats(target)
        scales=(self.out_norm.var.sqrt().clamp_min(1e-4),self.telemetry_norm.var.sqrt().clamp_min(1e-3))
        errors=self._rollout_errors(w,field,default_q,scales,target)
        valid=w['valid'];den=valid.sum().clamp_min(1)
        terms={name:(errors[name]*valid).sum()/den for name in ('state','pose','contact','telemetry','consistency')}
        result=terms['state']+terms['pose']+terms['contact']+terms['telemetry']+.1*terms['consistency']
        if target is not None:
            # Every valid control transition has its own current-image label.
            terms['latent']=(errors['latent']*target.refresh).sum()/target.refresh.sum().clamp_min(1)
            reconstruction=self.latent_decoder(encoder(target.reconstruction_noisy))
            terms['reconstruction']=F.l1_loss(reconstruction,target.reconstruction_clean)
            result=result+terms['latent']+terms['reconstruction']
        return (result,terms) if return_terms else result
