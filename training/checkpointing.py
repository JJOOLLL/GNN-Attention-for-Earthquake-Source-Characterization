"""Epoch-boundary checkpoints without consuming training random numbers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


TRAINING_FIELDS = (
    'epoch', 'best_val_loss', 'no_improve_count', 'train_flag', 'lr_change',
    'num_epochs', 'no_improve_epoch', 'train_loss_list', 'val_loss_list',
)


def _previous_path(path):
    return Path(path).with_suffix('.previous.pth')


def atomic_torch_save(obj, path, rotate=False):
    """Write and fsync a same-directory temporary file, then replace atomically.

    With rotation, only the checkpoint-owned ``*.previous.pth`` is replaced.
    An interrupted replacement leaves either the current or previous generation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = _previous_path(path)
    if rotate and previous == path:
        raise ValueError('Checkpoint name must not end in .previous.pth')
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w+b', dir=path.parent, prefix='.' + path.name + '.',
            suffix='.tmp', delete=False,
        ) as stream:
            temp = Path(stream.name)
            torch.save(obj, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if rotate and path.exists():
            # Do not replace a useful previous generation with a corrupt current one.
            try:
                prior = torch.load(path, map_location='cpu', weights_only=True)
                _check_finite_tensors(prior, 'previous_generation')
            except Exception:
                # The exception is intentionally confined to checkpoint readability.
                # The new, fully written temporary file can still become current.
                pass
            else:
                os.replace(path, previous)
        os.replace(temp, path)
        temp = None
    finally:
        if temp is not None:
            # This is the uniquely named temporary file created by this call only.
            temp.unlink(missing_ok=True)
    return path


def capture_rng():
    """Return RNG states containing only weights_only-safe built-in types/tensors."""
    np_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {
            'algorithm': np_state[0],
            'keys': np_state[1].tolist(),
            'position': int(np_state[2]),
            'has_gauss': int(np_state[3]),
            'cached_gaussian': float(np_state[4]),
        },
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    cuda_states = state['torch_cuda']
    if cuda_states and (
        not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()
    ):
        raise RuntimeError('CUDA device count differs from checkpoint RNG state')
    np_state = state['numpy']
    random.setstate(state['python'])
    np.random.set_state((
        np_state['algorithm'], np.asarray(np_state['keys'], dtype=np.uint32),
        np_state['position'], np_state['has_gauss'], np_state['cached_gaussian'],
    ))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def state_digest(model):
    """Keep the existing run_trace.py digest convention for historical checks."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), list(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_digest(state):
    """Same digest as the historical helper, with safe NumPy-state storage."""
    digest = hashlib.sha256()
    digest.update(repr(state['python']).encode())
    n = state['numpy']
    digest.update(repr((n['algorithm'], n['position'], n['has_gauss'], n['cached_gaussian'])).encode())
    digest.update(np.asarray(n['keys'], dtype=np.uint32).tobytes())
    digest.update(state['torch_cpu'].cpu().numpy().tobytes())
    for item in state['torch_cuda']:
        digest.update(item.cpu().numpy().tobytes())
    return digest.hexdigest()


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Unsupported checkpoint value: ' + type(value).__name__)


def _provenance_json(provenance):
    return json.dumps(provenance, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _check_finite_tensors(value, label):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise ValueError('Nonfinite checkpoint tensor: ' + label)
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite_tensors(item, label + '.' + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _check_finite_tensors(item, label + '.' + str(index))


def save_training_state(path, ns, provenance, best_state):
    before = capture_rng()
    payload = {
        'format_version': 1,
        'provenance': json.loads(_provenance_json(provenance)),
        'model': _cpu_copy(ns['model'].state_dict()),
        'optimizer': _cpu_copy(ns['optimizer'].state_dict()),
        'training': {key: _cpu_copy(ns[key]) for key in TRAINING_FIELDS},
        'rng': before,
        'best_state': _cpu_copy(best_state),
    }
    result = atomic_torch_save(payload, path, rotate=True)
    if rng_digest(capture_rng()) != rng_digest(before):
        restore_rng(before)
        raise RuntimeError('Checkpoint saving unexpectedly consumed training RNG')
    return result


def restore_training_state(path, ns, provenance):
    """Restore a valid generation; reject a provenance mismatch without fallback.

    Recreate the original datasets, model and optimizer before calling this.
    RNG restoration is the final operation before returning the best weights.
    """
    path = Path(path)
    errors = []
    payload = None
    for candidate in (path, _previous_path(path)):
        try:
            loaded = torch.load(candidate, map_location='cpu', weights_only=True)
        except Exception as error:
            errors.append(f'{candidate.name}: {type(error).__name__}: {error}')
            continue
        if not isinstance(loaded, dict) or loaded.get('format_version') != 1:
            raise ValueError('Unsupported training checkpoint format: ' + str(candidate))
        if _provenance_json(loaded.get('provenance')) != _provenance_json(provenance):
            raise ValueError('Checkpoint provenance mismatch: ' + str(candidate))
        if set(loaded['training']) != set(TRAINING_FIELDS):
            raise ValueError('Incomplete training state: ' + str(candidate))
        try:
            for key in ('model', 'optimizer', 'best_state'):
                _check_finite_tensors(loaded[key], key)
        except ValueError as error:
            errors.append(f'{candidate.name}: {error}')
            continue
        payload = loaded
        break
    if payload is None:
        raise RuntimeError('No readable training checkpoint. ' + '; '.join(errors))
    ns['model'].load_state_dict(payload['model'], strict=True)
    ns['optimizer'].load_state_dict(payload['optimizer'])
    ns.update(payload['training'])
    best_state = payload['best_state']
    restore_rng(payload['rng'])
    return best_state
