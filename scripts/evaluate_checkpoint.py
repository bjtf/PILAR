#!/usr/bin/env python3
"""Evaluate a final PILAR export without fitting normalization or selecting an epoch."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from lipnet.inference import PilarDetector
from lipnet.checkpoint import sha256
from lipnet.data import visa_split
from lipnet.metrics import roc_auc, pr_auc, EVALUATOR_VERSION
from lipnet.pro_metrics import aupro_at_fpr
from scripts.reproduce_paper import atomic_json
from scripts.run_mvtec_patch_lipnet import _test_images, _load_rgb, _image_to_tensor, _load_mask, _resize_array


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',choices=['mvtec','visa'],required=True)
    p.add_argument('--category',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    p.add_argument('--batch-size',type=int,default=1)
    a=p.parse_args()
    if a.batch_size<1:p.error('batch-size must be positive')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError('Use an empty output directory')
    model=PilarDetector(a.checkpoint,a.device)
    if getattr(model.args,'dataset','mvtec')!=a.dataset:
        raise ValueError('Checkpoint dataset mismatch')
    if not a.checkpoint.name.startswith(f'model_{a.category}_'):
        raise ValueError('Checkpoint/category mismatch')
    items=visa_split(a.data_root)[a.category]['test'] if a.dataset=='visa' else _test_images(a.data_root/a.category)
    images=[];scores=[];maps=[];masks=[];labels=[]
    for start in range(0,len(items),a.batch_size):
        rows=items[start:start+a.batch_size]
        rgb=torch.stack([_image_to_tensor(_load_rgb(x.path,model.args.image_resize)) for x in rows])
        batch_scores,batch_maps=model.predict(rgb)
        scores.extend(batch_scores);maps.extend(batch_maps)
        for x,s in zip(rows,batch_scores):
            mask=_load_mask(x.mask_path,(model.args.image_resize,)*2)
            mask=_resize_array((mask*255).astype(np.uint8),model.args.pro_eval_size,Image.Resampling.NEAREST)>0
            masks.append(mask);labels.append(x.label)
            images.append({'category':a.category,'image':str(x.path.relative_to(a.data_root/a.category)),
                           'defect_type':x.defect_type,'label':x.label,'score':float(s)})
    pro,curve,clipped=aupro_at_fpr(maps,masks,model.args.fpr_limit,model.args.num_thresholds,model.args.connectivity)
    metrics={'image_auc':roc_auc(np.asarray(scores),np.asarray(labels)),
             'image_ap':pr_auc(np.asarray(scores),np.asarray(labels)),
             'pixel_auc':roc_auc(np.asarray(maps).reshape(-1),np.asarray(masks).reshape(-1)),
             'aupro_30':pro['aupro']}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'image_scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(images[0]));w.writeheader();w.writerows(images)
    np.savez_compressed(a.output_dir/'maps.npz',heat_maps=maps,masks=masks,scores=scores,labels=labels,
                        images=[x['image'] for x in images])
    atomic_json(a.output_dir/'metrics.json',{'dataset':a.dataset,'category':a.category,'checkpoint_sha256':sha256(a.checkpoint),
                'evaluator':EVALUATOR_VERSION,'device':a.device,'batch_size':a.batch_size,'metrics':metrics,
                'pro_curve':curve,'pro_curve_clipped':clipped})
    print(metrics)

if __name__=='__main__':main()
