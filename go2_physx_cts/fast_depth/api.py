"""Accepted compact refiner + unchanged geometric projection and frozen noise."""
from pathlib import Path
import torch
from .model import SurfaceRefiner,ModelConfig
from .source import import_package,sha256,package_fingerprints

class FastDepthGenerator:
    def __init__(self,weights,depth_package,device='cuda:0',chunk_size=32,compile_mode='default',precision='fp32',require_accepted=True):
        package=import_package(depth_package);ck=torch.load(weights,map_location='cpu',weights_only=False)
        if ck.get('schema')!='surface_depth_v2':raise ValueError('Expected compact surface-depth checkpoint')
        acceptance=ck.get('acceptance',{})
        if require_accepted and (not acceptance.get('passed') or precision not in acceptance.get('accepted_precisions',[])):
            raise ValueError('This observer/precision has not passed independent acceptance')
        actual=sha256(Path(depth_package)/'go2_depth_input/assets/weights.pt')
        if ck['source_weights_sha256']!=actual or ck['source_code_sha256']!=package_fingerprints(depth_package):
            raise ValueError('Pinned geometry/noise source has changed')
        self.engine=package.DepthInputGenerator(device=device,chunk_size=chunk_size,compile_mode='off',precision=precision)
        model=SurfaceRefiner(ModelConfig(**ck['model_config'])).to(device)
        model.load_state_dict(ck['model']);model.install(self.engine.core.geometry)
        self.engine.core.eval().requires_grad_(False).to(memory_format=torch.channels_last)
        self.engine.compile_mode=compile_mode
        if compile_mode not in ('off','default','max-autotune-no-cudagraphs'):raise ValueError('Unsupported BPTT compile mode')
        self.engine._fn=self.engine.core if compile_mode=='off' else torch.compile(self.engine.core,mode=compile_mode,dynamic=False,fullgraph=True)
        self.metadata={'schema':'surface_depth_v2','checkpoint_sha256':sha256(weights),'source_weights_sha256':actual,
            'precision':precision,'frozen':True,'acceptance':acceptance,
            'geometry':'original full-resolution 32-sample surface projection + compact learned refiner',
            'noise':'unchanged original frozen noise network and explicit keys',
            'refiner_parameters':sum(p.numel() for p in model.parameters()),'model_config':model.configuration()}
    def __getattr__(self,name):return getattr(self.engine,name)
