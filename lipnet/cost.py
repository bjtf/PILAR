"""Executed weighted-operator cost; excludes non-affine and data movement work.

Dense convolutions/linear layers use 1 MAC = 2 FLOPs. PILAR's shared spatial
weight multiplication and window reductions are counted as actually written,
without charging repeated multiplications for overlapping receptive fields.
This scope is an arithmetic descriptor, not a prediction of measured runtime.
"""
from collections import Counter
import torch
from lipnet.model import LIPNet2DLayer
from lipnet.inhibition import LateralInhibition

SCOPE = ('Executed Conv2d/Linear and PILAR spatial weighted reductions/inhibition; '
         '1 dense MAC = 2 FLOPs, each spatial multiply/add = 1 FLOP. Includes '
         'dense zero/padded convolution terms and actual padded spatial multiplies. '
         'Excludes bias, activation, normalization, pooling, interpolation, radial '
         'kernel construction, residual scoring, sorting and CPU postprocessing. '
         'Not total pipeline FLOPs or an estimate of latency/energy.')

@torch.no_grad()
def weighted_operator_flops(modules, detector, rgb):
    counts = Counter()
    handles = []
    seen = set()
    def hook(module, inputs, output):
        x = inputs[0]
        if isinstance(module, torch.nn.Conv2d):
            counts['conv2d'] += 2 * output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, torch.nn.Linear):
            counts['linear'] += 2 * output.numel() * module.in_features
        elif isinstance(module, LIPNet2DLayer):
            batch = x.shape[0]
            counts['spatial_weight_multiply'] += batch * module.channels * (module.in_size + module.pad_bottom) * (module.in_size + module.pad_right)
            counts['spatial_window_sum'] += batch * module.channels * module.base_out_size ** 2 * (module.rf ** 2 - 1)
        elif isinstance(module, LateralInhibition):
            k = 2 * module.h + 1
            if min(x.shape[-2:]) >= k:
                counts['inhibition_conv2d'] += 2 * output.numel() * k * k
    try:
        for container in modules:
            for module in container.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, LIPNet2DLayer, LateralInhibition)):
                    handles.append(module.register_forward_hook(hook))
        detector.predict(rgb)
    finally:
        for handle in handles:
            handle.remove()
    return {'per_image': sum(counts.values()) / len(rgb),
            'breakdown_per_image': {key: value / len(rgb) for key, value in sorted(counts.items())},
            'scope': SCOPE}
