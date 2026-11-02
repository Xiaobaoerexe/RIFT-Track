"""Exact batching of known Replay inputs; no network or persistent feature cache.

Only the GRU requires ordered time execution. Every invocation uses the current
encoder weights. A segment's first frame is always encoded anew, so crossing from
no-grad burn-in into a supervised segment cannot reuse a detached CNN feature.
"""
import torch
from torch.utils.checkpoint import checkpoint

ENCODING_CHUNK=1024


def batched_apply(function,values,training=False,chunk_size=ENCODING_CHUNK):
    """Bound activation memory while retaining the same samplewise function."""
    pieces=[]
    for part in values.split(chunk_size):
        value=checkpoint(function,part,use_reentrant=False) if training and torch.is_grad_enabled() else function(part)
        pieces.append(value)
    return torch.cat(pieces,0)


def depth_sequence_features(student,depth,sequence,start,stop,training=False,chunk_size=ENCODING_CHUNK):
    """Current-encoder features[B,T,128], reusing each held image within a segment.

    Replay poses and noise keys are known inputs, so frozen image generation is
    inference-only here. This helper is NEVER used for future imagined poses.
    """
    b=sequence['obs'].shape[0];length=stop-start
    if length<=0:return sequence['obs'].new_empty((b,0,128),dtype=torch.float32)
    keys=torch.cat([sequence[k][:,start:stop] for k in ('depth_frame_index','episode','depth_scene_ids')],-1)
    refresh=torch.ones((b,length),dtype=torch.bool,device=keys.device)
    refresh[:,1:]=(keys[:,1:]!=keys[:,:-1]).any(-1)
    capture_rows=refresh.flatten().nonzero().flatten()
    # Every row begins with a fresh capture; flat cumulative indexing therefore
    # never carries an image across independent trajectories.
    feature_indices=refresh.flatten().long().cumsum(0)-1
    def select(name):
        value=sequence[name][:,start:stop]
        return value.reshape(b*length,*value.shape[2:])[capture_rows]
    pose=select('depth_pose');scene=select('depth_scene_ids').reshape(-1)
    frame=select('depth_frame_index').reshape(-1);episode=select('episode').reshape(-1)
    trajectories=sequence['trajectory_ids'].reshape(b,1).expand(b,length).reshape(-1)[capture_rows]
    # Transfer small capture metadata once. Subsequent CPU gathers and H2D
    # transfers can overlap the preceding chunk's GPU CNN work; no per-chunk
    # metadata .cpu() synchronization is needed.
    capture_keys=depth.images.keys(pose,scene,frame,episode,trajectories)
    features=[]
    for lo in range(0,len(capture_rows),chunk_size):
        part=slice(lo,lo+chunk_size)
        with torch.no_grad():
            images=depth.captured(pose[part],scene[part],frame[part],episode[part],trajectories[part],capture_keys=capture_keys[part])
        feature=checkpoint(student.encode_depth,images,use_reentrant=False) if training and torch.is_grad_enabled() else student.encode_depth(images)
        features.append(feature)
    encoded=torch.cat(features,0)
    return encoded[feature_indices].reshape(b,length,128)
