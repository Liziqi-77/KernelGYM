import random

import numpy as np
import torch

from verl.utils.device import get_device_name


def save_random_states():
    device_name = get_device_name()
    if device_name == 'cuda':
        torch_device_state = torch.cuda.get_rng_state_all()
    elif device_name == 'npu':
        torch_device_state = torch.npu.get_rng_state_all()
    else:
        raise NotImplementedError(f"Unsupported device: {device_name}")
    
    rng_states = {
        'torch_cpu': torch.get_rng_state(),
        'torch_device': torch_device_state,
        'random': random.getstate(),
        'numpy': np.random.get_state(),
    }
    return rng_states


def set_global_seed(seed):
    device_name = get_device_name()
    torch.manual_seed(seed)
    if device_name == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif device_name == 'npu':
        torch.npu.manual_seed_all(seed)
    else:
        raise NotImplementedError(f"Unsupported device: {device_name}")
    random.seed(seed)
    np.random.seed(seed)


def set_random_states(rng_states):
    if rng_states is None:
        set_global_seed(42)
    else:
        device_name = get_device_name()
        torch.set_rng_state(rng_states['torch_cpu'])
        if device_name == 'cuda':
            torch.cuda.set_rng_state_all(rng_states['torch_device'])
        elif device_name == 'npu':
            torch.npu.set_rng_state_all(rng_states['torch_device'])
        else:
            raise NotImplementedError(f"Unsupported device: {device_name}")
        random.setstate(rng_states['random'])
        np.random.set_state(rng_states['numpy'])
