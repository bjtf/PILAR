"""Common inference contract: resized CPU RGB [0,1] -> CPU scores/maps."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Protocol
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class Detector(Protocol):
    def predict(self, rgb: torch.Tensor) -> tuple[np.ndarray,np.ndarray]: ...

class PilarDetector:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        from lipnet.checkpoint import SCHEMA
        from scripts.run_mvtec_feature_lipnet_fcn_sweep import ResNetMultiFeatureExtractor, LayerStats
        from scripts.run_mvtec_feature_lipnet import _make_feature_model
        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if state['schema']!=SCHEMA or state['method']!='pilar':raise ValueError('Wrong checkpoint schema/method')
        self.args=argparse.Namespace(**state['config'])
        self.device=torch.device(device)
        self.backbone=ResNetMultiFeatureExtractor(self.args.backbone,state['models'],'none').to(self.device).eval()
        self.backbone.load_state_dict(state['backbone'],strict=True)
        self.models={}
        with torch.no_grad():
            shapes=self.backbone(torch.zeros(1,3,self.args.image_resize,self.args.image_resize,device=self.device))
        for layer,weights in state['models'].items():
            m=_make_feature_model(self.args,tuple(shapes[layer].shape[1:])).to(self.device).eval()
            m.load_state_dict(weights,strict=True);self.models[layer]=m
        self.stats={k:LayerStats(**v) for k,v in state['stats'].items()}

    @torch.no_grad()
    def predict(self, rgb):
        from scripts.run_mvtec_feature_lipnet_fcn_sweep import _normalize_feature,_feature_loss_and_map,_apply_score_norm,_aggregate_map_score
        from scripts.run_mvtec_patch_lipnet import _resize_array
        if rgb.device.type!='cpu' or rgb.dtype!=torch.float32 or rgb.ndim!=4:
            raise ValueError('Expected CPU float32 [B,3,H,W]')
        if tuple(rgb.shape[1:])!=(3,self.args.image_resize,self.args.image_resize):
            raise ValueError('Input size mismatch')
        features=self.backbone(rgb.to(self.device))
        maps=[]
        for layer,model in self.models.items():
            feature=_normalize_feature(features[layer],self.args.feature_normalize)
            _,loc=_feature_loss_and_map(model,feature,self.args.l1_weight,0.,0.)
            maps.append(F.interpolate(_apply_score_norm(loc,self.stats[layer]).unsqueeze(1),size=(self.args.image_resize,)*2,mode='bilinear',align_corners=False))
        heat=torch.stack(maps).mean(0).squeeze(1).cpu().numpy().astype(np.float32)
        scores=np.array([_aggregate_map_score(h,self.args.image_agg,self.args.image_topk_frac) for h in heat])
        size=self.args.pro_eval_size
        evaluated=np.stack([_resize_array(h,size,Image.Resampling.BILINEAR).astype(np.float32) if size>0 else h for h in heat])
        return scores,evaluated
