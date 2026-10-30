"""Environment identity and a shared lock for this checkout's GPU workflows."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import platform
import os
import sys
import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version


def environment_identity(device):
    device = torch.device(device)
    accelerator = None
    if device.type == 'cuda':
        accelerator = torch.cuda.get_device_name(device)
    elif device.type == 'mps':
        accelerator = platform.machine()
    return {'python': sys.version, 'platform': platform.platform(), 'host': platform.node(),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'numpy': np.__version__, 'pillow': pillow_version,
            'device': str(device), 'accelerator': accelerator,
            'mps_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()}


@contextmanager
def device_lock(device):
    """Coordinate the paper runner and timing command; external programs are independent."""
    if str(device) == 'cpu':
        yield
        return
    path = Path(__file__).resolve().parents[1] / '.accelerator.lock'
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another PILAR paper training/timing command holds the accelerator lock') from exc
        yield
