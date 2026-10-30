#!/usr/bin/env python3
"""Same-device timing from CPU RGB through CPU image score and evaluation map."""
from pathlib import Path
import argparse,datetime,fcntl,json,os,platform,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torchvision
SCHEMA = "pilar-paper-timing-v1"
from lipnet.checkpoint import sha256
from lipnet.cost import weighted_operator_flops
from lipnet.inference import PilarDetector
from lipnet.runtime import device_lock
from scripts.run_mvtec_patch_lipnet import _test_images,_load_rgb,_image_to_tensor,MVTEC_CATEGORIES

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--data-root',type=Path,required=True)
 p.add_argument('--device',choices=['mps','cuda','cpu'],default='mps')
 p.add_argument('--throughput-batch',type=int,choices=[8,16],default=16)
 p.add_argument('--diagnostic',action='store_true')
 a=p.parse_args()
 if a.device=='mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1'):
    raise RuntimeError('Real MPS without fallback required')
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 def synchronize():
    if a.device=='mps':torch.mps.synchronize()
    elif a.device=='cuda':torch.cuda.synchronize()

 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 with device_lock(a.device):
  # The paper runner uses the same lock; also check known standalone training entry points.
  processes=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
  active=[x for x in processes if 'python' in x.lower() and any(n in x for n in ['run_mvtec_feature_lipnet_fcn_arch_sweep.py','reproduce_paper.py','run_cifar10_feature_lipnet_fcn.py','oneclass_auc_per_class.py'])]
  if active:raise RuntimeError('Training process active: '+repr(active))
  if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a new empty timing output directory; previous measurements are preserved')
  a.output.mkdir(parents=True,exist_ok=True)
  warm=2 if a.diagnostic else 50;latency_calls=3 if a.diagnostic else 200;batch_calls=2 if a.diagnostic else 100;blocks=1 if a.diagnostic else 5
  categories=MVTEC_CATEGORIES[:1] if a.diagnostic else MVTEC_CATEGORIES
  code_files=sorted(set((ROOT/'lipnet').glob('*.py'))|set((ROOT/'scripts').glob('*.py'))|{Path(__file__).resolve()})
  source_hashes={str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p.resolve()):sha256(p) for p in code_files}
  metadata={'training_epochs':10,'schema':SCHEMA,'torch_num_threads':torch.get_num_threads(),'torch_num_interop_threads':torch.get_num_interop_threads(),'cpu_thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},'python':platform.python_version(),'torchvision':torchvision.__version__,'input_size':512,'output_size':256,'mps_fallback':False,'source_hashes':source_hashes,'command':sys.argv,'method':'pilar','host':platform.node(),'platform':platform.platform(),'machine':subprocess.check_output(['system_profiler','SPHardwareDataType','SPDisplaysDataType'],text=True) if sys.platform=='darwin' else platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'precision':'FP32','tf32':False,'device':a.device.upper(),'seed':42,'latency_batch':1,'throughput_batch':a.throughput_batch,'warmup_calls':warm,'blocks':blocks,'latency_calls_per_block':latency_calls,'throughput_batches_per_block':batch_calls,'scope':'CPU float32 resized RGB -> normalize, transfer, backbone, detector, image scoring, map postprocess -> CPU score/map256; excludes disk and metrics','synchronization':'before and after latency call; at throughput block boundaries (CPU outputs also synchronize)','diagnostic':a.diagnostic,'started':datetime.datetime.now().astimezone().isoformat(),'categories':{}}
  (a.output/'protocol.json').write_text(json.dumps(metadata,indent=2))
  for category in categories:
     found=list(a.run.glob(f'details/**/model_{category}_*.pt'))
     if len(found)!=1:raise RuntimeError(f'Ambiguous/missing checkpoint: {category}')
     checkpoint=found[0];detector=PilarDetector(checkpoint,device=a.device)
     if detector.args.seed!=42 or detector.args.epochs!=10 or detector.args.image_resize!=512 or detector.args.pro_eval_size!=256 or getattr(detector.args,'dataset','mvtec')!='mvtec':
         raise ValueError('Expected final paper MVTec seed42 checkpoint,10epochs,512input')
     expected={'feature_layers':'layer2,layer3','feature_encoder_channels':'64,64','feature_decoder_channels':'128,64','feature_inhibition':True,'feature_normalize':True,'image_agg':'topk_mean','image_topk_frac':0.01,'l1_weight':0.0}
     if any(getattr(detector.args,key,None)!=value for key,value in expected.items()):
         raise ValueError('Expected the main paper detector, not an ablation')
     if detector.args.hard_loss_settings != ['0.10:1.0']:
         raise ValueError('Unexpected hard residual training configuration')
     modules=[detector.backbone,*detector.models.values()]
     # Same deterministic paths for all methods; selection does not use scores/labels.
     items=sorted(_test_images(a.data_root/category),key=lambda x:str(x.path))[:16]
     if len(items)!=16:raise ValueError('Need the same 16 images for all timing batches')
     rgb=torch.stack([_image_to_tensor(_load_rgb(x.path,512)) for x in items])
     lat=[];lat_blocks=[];through=[];failure=None
     def call(x):
         scores,maps=detector.predict(x)
         if scores.shape!=(len(x),) or maps.shape!=(len(x),256,256):raise ValueError('Common output contract failed')
     for i in range(warm):call(rgb[i%len(rgb):i%len(rgb)+1])
     for block in range(blocks):
         block_start=time.perf_counter()
         for i in range(latency_calls):
             x=rgb[i%len(rgb):i%len(rgb)+1];synchronize();start=time.perf_counter();call(x);synchronize();lat.append({'block':block,'iteration':i,'seconds':time.perf_counter()-start})
         lat_blocks.append({'block':block,'calls':latency_calls,'seconds':time.perf_counter()-block_start})
     arithmetic=weighted_operator_flops(modules,detector,rgb[:1])
     b=a.throughput_batch;x=rgb[:b]
     try:
         for i in range(warm):call(x)
         for block in range(blocks):
             synchronize();start=time.perf_counter()
             for i in range(batch_calls):call(x)
             synchronize();elapsed=time.perf_counter()-start
             through.append({'block':block,'batches':batch_calls,'images':b*batch_calls,'seconds':elapsed,'images_per_second':b*batch_calls/elapsed})
     except RuntimeError as exc:
         if 'out of memory' not in str(exc).lower():raise
         failure=str(exc)
     params={id(p):p for m in modules for p in m.parameters()};buffers={id(p):p for m in modules for p in m.buffers()}
     data={'category':category,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'image_paths':[str(i.path.relative_to(a.data_root)) for i in items],'image_sha256':[sha256(i.path) for i in items],'latency_calls':lat,'latency_blocks':lat_blocks,'throughput_blocks':through,'throughput_error':failure,'latency_ms':{'median':float(np.median([r['seconds'] for r in lat])*1000),'p95':float(np.quantile([r['seconds'] for r in lat],.95)*1000),'mean':float(np.mean([r['seconds'] for r in lat])*1000)},'throughput_images_per_second':sum(r['images'] for r in through)/sum(r['seconds'] for r in through) if through else None,'parameters_trainable':sum(p.numel() for p in params.values() if p.requires_grad),'parameters_total_retained':sum(p.numel() for p in params.values()),'tensor_state_bytes':sum(p.numel()*p.element_size() for p in list(params.values())+list(buffers.values())),'serialized_inference_checkpoint_bytes':checkpoint.stat().st_size,'weighted_operator_flops':arithmetic,'diagnostic':a.diagnostic}
     (a.output/(category+'.json')).write_text(json.dumps(data,indent=2));print(category,data['latency_ms'],data['throughput_images_per_second'],flush=True)
     del detector,modules,params,buffers
     if a.device=="mps":torch.mps.empty_cache()
     elif a.device=="cuda":torch.cuda.empty_cache()
     if failure:raise RuntimeError('Batch capacity failure recorded. Run batch 8 for ALL methods before comparison.')
  metadata['reference_backend']=a.device=='mps'
  metadata['completed']=datetime.datetime.now().astimezone().isoformat();metadata['categories']=categories;(a.output/'protocol.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
