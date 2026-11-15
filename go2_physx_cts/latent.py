"""Every-control-step camera-latent supervision, adapted from InstinctMJ's latent rollout.

Reference: project-instinct/InstinctMJ@03563ee02665f327425106372264070315673748,
rl/supermimic/{runner,networks}.py (CC BY-NC 4.0; see licenses/).
50 Hz camera and controller; eight autoregressive supervised transitions.
The latent denotes the current image; every valid transition is supervised.
No permanently cached trainable feature, target encoder, or imagined renderer.
"""
from dataclasses import dataclass
import torch
from torch import nn
from . import state as st
from .precision import network_autocast
from .sequence_execution import batched_apply

LATENT_DIM = 128
class LatentDecoder(nn.Module):
    """Training-only 128-vector to clean normalized 48x64 depth reconstruction."""
    def __init__(self, latent_dim=LATENT_DIM, mixed_precision=True):
        super().__init__(); self.mixed_precision=mixed_precision
        self.fc=nn.Sequential(nn.Linear(latent_dim,32*6*8),nn.ELU())
        self.dec=nn.Sequential(
            nn.ConvTranspose2d(32,32,4,2,1),nn.ELU(),
            nn.ConvTranspose2d(32,16,4,2,1),nn.ELU(),
            nn.ConvTranspose2d(16,8,4,2,1),nn.ELU(),
            nn.Conv2d(8,1,3,1,1))
    def forward(self,z):
        with network_autocast(z,self.mixed_precision):
            image=self.dec(self.fc(z).reshape(-1,32,6,8))
        return image.float().sigmoid()


@dataclass
class LatentTargets:
    initial: torch.Tensor
    targets: torch.Tensor
    refresh: torch.Tensor
    increments: torch.Tensor
    reconstruction_noisy: torch.Tensor
    reconstruction_clean: torch.Tensor


@torch.no_grad()
def prepare_latent_targets(window, depth, encoder, period=1, reconstruction_images=512):
    """Re-encode immutable real captures with the current CNN, outside BPTT.

    Terminal targets use pose_next/frame+1 before reset, never the next episode.
    ``valid`` masks transitions after reset, but includes the terminal transition.
    Only refresh images are materialized, not B*K copies of held images.
    """
    w=window; batch,steps=w['a'].shape[:2]
    def sequence(name):
        return w[name].reshape(batch,steps,-1)
    trajectory=w.get('trajectory_ids',w.get('env'))
    if trajectory is None:raise ValueError('Latent targets require trajectory_ids')
    if trajectory.numel()==batch:trajectory=trajectory.reshape(batch,1).expand(-1,steps)
    else:trajectory=trajectory.reshape(batch,steps)
    episode=sequence('episode')[...,0].long()
    frame=sequence('frame_index')[...,0].long()
    cap_pose=sequence('depth_pose')[:,0]
    cap_scene=sequence('depth_scene_ids')[:,0,0].long()
    cap_frame=sequence('depth_frame_index')[:,0,0].long()
    first=depth.captured(cap_pose,cap_scene,cap_frame,episode[:,0],trajectory[:,0])
    initial=batched_apply(encoder,first).float()
    refresh=w['valid'].reshape(batch,steps).bool()
    rows,cols=refresh.nonzero(as_tuple=True)
    targets=initial.new_zeros((batch,steps,LATENT_DIM))
    if len(rows):
        next_pose=w['pose_next'][rows,cols]
        next_scene=depth.field.scene_ids(next_pose)
        next_frame=frame[rows,cols]+1
        noisy=depth.captured(next_pose,next_scene,next_frame,episode[rows,cols],trajectory[rows,cols])
        # Frozen image cache is FP16; captured() defines the canonical quantization.
        targets[rows,cols]=batched_apply(encoder,noisy).float()
        nr=min(reconstruction_images,len(rows))
        take=torch.randperm(len(rows),device=rows.device)[:nr]
        _,clean=depth.paired(next_pose[take],next_scene[take],next_frame[take],episode[rows[take],cols[take]],trajectory[rows[take],cols[take]])
        recon_noisy=noisy[take]
    else:
        # Small diagnostic windows need not contain a refresh; the CNN can still
        # reconstruct a real captured image without inventing a latent label.
        nr=min(reconstruction_images,batch)
        take=torch.randperm(batch,device=initial.device)[:nr]
        _,clean=depth.paired(cap_pose[take],cap_scene[take],cap_frame[take],episode[take,0],trajectory[take,0])
        recon_noisy=first[take]
    held=initial
    increments=[]
    for j in range(steps):
        increments.append(targets[:,j]-held)
        held=torch.where(refresh[:,j,None],targets[:,j],held)
    return LatentTargets(initial,targets,refresh,torch.stack(increments,1),recon_noisy,clean.float()/6.)
