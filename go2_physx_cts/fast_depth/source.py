"""Access the pinned observation package without modifying it."""
from contextlib import contextmanager
from pathlib import Path
import hashlib,sys
import torch


def import_package(path):
    path=Path(path).resolve()
    if not (path/'go2_depth_input').is_dir():raise FileNotFoundError(path)
    if str(path) not in sys.path:sys.path.insert(0,str(path))
    import go2_depth_input
    if Path(go2_depth_input.__file__).resolve().parent.parent!=path:
        raise RuntimeError('A different go2_depth_input package is already imported')
    return go2_depth_input


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_fingerprints(path):
    root=Path(path)/'go2_depth_input'
    files=('api.py','core.py','vendor/geometry_reference.py','vendor/noise_reference.py','vendor/camera.py','assets/source_lock.json')
    return {name:sha256(root/name) for name in files if (root/name).is_file()}


@contextmanager
def ieee_fp32(device):
    if torch.device(device).type!='cuda':
        yield;return
    old=torch.backends.cuda.matmul.allow_tf32
    with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                                  benchmark=torch.backends.cudnn.benchmark,
                                  benchmark_limit=torch.backends.cudnn.benchmark_limit,
                                  deterministic=torch.backends.cudnn.deterministic,
                                  allow_tf32=False):
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            yield
        finally:torch.backends.cuda.matmul.allow_tf32=old


def tangent_pose(position,rotation,delta):
    # Delta is a world-frame translation and SO(3) tangent in radians, not an
    # arbitrary perturbation of nine rotation-matrix elements.
    # A safe Taylor branch is also needed for Sobolev training's mixed second
    # derivatives. Differentiating norm(rotvec) at zero can poison those with
    # NaNs even when the first-order pose derivative looks finite.
    w=delta[:,3:];x,y,z=w.unbind(-1);zero=torch.zeros_like(x)
    k=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
    theta2=w.square().sum(-1);safe=theta2.clamp_min(1e-6).sqrt()
    a=torch.where(theta2<1e-6,1-theta2/6+theta2.square()/120,safe.sin()/safe)
    b=torch.where(theta2<1e-6,.5-theta2/24+theta2.square()/720,(1-safe.cos())/safe.square())
    dr=torch.eye(3,device=position.device,dtype=position.dtype)[None]+a[:,None,None]*k+b[:,None,None]*(k@k)
    return position+delta[:,:3],dr@rotation


def soft_image(depth,logit):return depth*logit.sigmoid()
