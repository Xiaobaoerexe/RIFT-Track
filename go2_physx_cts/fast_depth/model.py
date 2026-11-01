"""Cached scene representation + full camera pose -> direct metric depth.

No ray marching, physical state prediction or noise learning is performed here.
The source terrain CNN is frozen; this module learns only scene compression and
the direct observation decoder. Compression is cached only after freezing it.
"""
from dataclasses import asdict,dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    scene_dim:int=1024
    pose_dim:int=128
    seed_channels:int=32
    position_frequencies:int=4


class DirectDepth(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or ModelConfig();c=self.config
        self.scene=nn.Sequential(nn.Conv2d(17,16,3,2,1),nn.SiLU(),
            nn.Conv2d(16,32,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d((8,8)),
            nn.Flatten(),nn.Linear(32*8*8,c.scene_dim),nn.LayerNorm(c.scene_dim))
        pose_input=12+6*c.position_frequencies
        self.pose=nn.Sequential(nn.Linear(pose_input,c.pose_dim),nn.SiLU(),nn.Linear(c.pose_dim,c.pose_dim),nn.SiLU())
        self.seed=nn.Sequential(nn.Linear(c.scene_dim+c.pose_dim,512),nn.SiLU(),nn.Linear(512,c.seed_channels*6*8))
        self.up1=nn.Sequential(nn.Conv2d(c.seed_channels,24,3,1,1),nn.SiLU())
        self.up2=nn.Sequential(nn.Conv2d(24,16,3,1,1),nn.SiLU())
        self.up3=nn.Sequential(nn.Conv2d(16,8,3,1,1),nn.SiLU())
        self.output=nn.Conv2d(8,2,1)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('frequency',torch.pi*2**torch.arange(c.position_frequencies,dtype=torch.float32),persistent=False)

    @staticmethod
    def scene_inputs(source_cache):
        # Each source channel has an explicit geometric meaning; compression
        # cannot silently become a scene-ID embedding or omit scene information.
        maps,local,context=source_cache[:3]
        return torch.cat((F.interpolate(maps[:,None],size=(64,64),mode='bilinear',align_corners=True)/6.,
            F.adaptive_avg_pool2d(local,(64,64)),context),1).float()

    def encode_scenes(self,source_features):return self.scene(source_features)

    def decode(self,embedding,position,rotation):
        p=position.float()/position.new_tensor([12.8,12.8,6.])
        phase=(p[:,:,None]*self.frequency).flatten(1)
        condition=torch.cat((p,rotation.float().flatten(1),phase.sin(),phase.cos()),-1)
        pose=self.pose(condition)
        image=self.seed(torch.cat((embedding,pose),-1)).reshape(-1,self.config.seed_channels,6,8)
        image=self.up1(F.interpolate(image,size=(12,16),mode='bilinear',align_corners=False))
        image=self.up2(F.interpolate(image,size=(24,32),mode='bilinear',align_corners=False))
        image=self.up3(F.interpolate(image,size=(48,64),mode='bilinear',align_corners=False))
        output=self.output(image).float()
        return .1+5.9*output[:,0].sigmoid(),output[:,1]

    def forward(self,source_features,scene_ids,position,rotation):
        return self.decode(self.encode_scenes(source_features)[scene_ids],position,rotation)

    def freeze(self):
        self.eval();self.requires_grad_(False);return self

    def configuration(self):return asdict(self.config)
