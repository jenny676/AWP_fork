import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random as pyrandom
import shutil
from pathlib import Path

from wideresnet import WideResNet
from preactresnet import PreActResNet18
from utils import *
from utils_awp import AdvWeightPerturb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mu = torch.tensor(cifar10_mean, dtype=torch.float32).view(3, 1, 1).to(device)
std = torch.tensor(cifar10_std, dtype=torch.float32).view(3, 1, 1).to(device)

# --- add near imports / top of file ---
import torch
import logging

logger = logging.getLogger(__name__)

def safe_torch_load(path, map_location=None):
    """
    Load a checkpoint with weights_only=False for PyTorch >=2.6 compatibility.
    Falls back to torch.load(path, map_location=...) if weights_only arg not supported.
    """
    try:
        # preferred: explicit full unpickle
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older torch that doesn't accept weights_only
        return torch.load(path, map_location=map_location)

# -------------------------
# Checkpoint helpers
# -------------------------
def _maybe_state_dict(obj):
    """Return state_dict() if available else None."""
    if obj is None:
        return None
    return obj.state_dict() if hasattr(obj, 'state_dict') else None

def save_full_checkpoint(path, model, opt, proxy=None, proxy_opt=None, awp=None,
                         scaler=None, scheduler=None, epoch=0, batch_idx=0,
                         best_test=-1.0, best_val=-1.0, train_subset_indices=None, logger=None, extra=None):
    """
    Atomic save of a full checkpoint including batch index so we can resume mid-epoch.
    """
    state = {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'best_test_robust_acc': best_test,
        'best_val_robust_acc': best_val,
        'rng_numpy': np.random.get_state(),
        'rng_python': pyrandom.getstate(),
        'rng_torch': torch.get_rng_state(),
        'train_subset_indices': None if train_subset_indices is None else list(train_subset_indices),
    }
    # optional components
    if proxy is not None:
        state['proxy_state'] = _maybe_state_dict(proxy)
    if proxy_opt is not None:
        state['proxy_opt_state'] = _maybe_state_dict(proxy_opt)
    if awp is not None and hasattr(awp, 'state_dict'):
        try:
            state['awp_state'] = awp.state_dict()
        except Exception:
            # fallback: store nothing if state_dict fails
            state['awp_state'] = None
    if scaler is not None and hasattr(scaler, 'state_dict'):
        state['scaler_state'] = scaler.state_dict()
    if scheduler is not None and hasattr(scheduler, 'state_dict'):
        state['scheduler_state'] = scheduler.state_dict()
    if extra is not None:
        state['extra'] = extra
    if torch.cuda.is_available():
        # store CUDA RNGs
        state['rng_cuda_all'] = torch.cuda.get_rng_state_all()
    # use existing atomic saver
    save_checkpoint(state, path, logger)


def load_full_checkpoint(path, model, opt, proxy=None, proxy_opt=None, awp=None,
                         scaler=None, scheduler=None, device='cpu'):
    """
    Load a checkpoint into model/optimizers. Returns (ckpt, start_epoch, resume_batch_idx).
    Loads to CPU first for safety.
    """
    ckpt = safe_torch_load(path, map_location='cpu')
    # load model (be permissive)
    if 'model_state' in ckpt:
        try:
            model.load_state_dict(ckpt['model_state'])
        except Exception as e:
            # try strict=False if shapes differ slightly
            try:
                model.load_state_dict(ckpt['model_state'], strict=False)
                print(f"Warning loading model_state with strict=False: {e}")
            except Exception as e2:
                print("Warning: failed to load model_state:", e2)

    if 'opt_state' in ckpt:
        try:
            opt.load_state_dict(ckpt['opt_state'])
        except Exception as e:
            print("Warning: could not load optimizer state:", e)

    # optional components
    if proxy is not None and 'proxy_state' in ckpt and ckpt['proxy_state'] is not None:
        try:
            proxy.load_state_dict(ckpt['proxy_state'])
        except Exception as e:
            print("Warning: could not load proxy state:", e)
    if proxy_opt is not None and 'proxy_opt_state' in ckpt and ckpt['proxy_opt_state'] is not None:
        try:
            proxy_opt.load_state_dict(ckpt['proxy_opt_state'])
        except Exception as e:
            print("Warning: could not load proxy optimizer state:", e)
    if awp is not None and 'awp_state' in ckpt and ckpt['awp_state'] is not None and hasattr(awp, 'load_state_dict'):
        try:
            awp.load_state_dict(ckpt['awp_state'])
        except Exception as e:
            print("Warning: could not load awp state:", e)
    if scaler is not None and 'scaler_state' in ckpt and ckpt['scaler_state'] is not None:
        try:
            scaler.load_state_dict(ckpt['scaler_state'])
        except Exception as e:
            print("Warning: could not load scaler state:", e)
    if scheduler is not None and 'scheduler_state' in ckpt and ckpt['scheduler_state'] is not None:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        except Exception as e:
            print("Warning: could not load scheduler state:", e)

    # restore RNGs robustly
    try:
        if 'rng_numpy' in ckpt: np.random.set_state(ckpt['rng_numpy'])
        if 'rng_python' in ckpt: pyrandom.setstate(ckpt['rng_python'])
        if 'rng_torch' in ckpt:
            rt = ckpt['rng_torch']
            if isinstance(rt, torch.Tensor):
                torch.set_rng_state(rt)
            else:
                # compatibility: attempt to convert to torch tensor
                try:
                    torch.set_rng_state(torch.tensor(rt, dtype=torch.uint8))
                except Exception:
                    pass
        if torch.cuda.is_available() and 'rng_cuda_all' in ckpt:
            cuda_states = []
            for s in ckpt['rng_cuda_all']:
                if isinstance(s, torch.Tensor):
                    cuda_states.append(s)
                else:
                    cuda_states.append(torch.tensor(s, dtype=torch.uint8))
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as e:
        print("Warning: restoring RNGs failed:", e)

    start_epoch = int(ckpt.get('epoch', 0))
    resume_batch_idx = int(ckpt.get('batch_idx', 0))
    return ckpt, start_epoch, resume_batch_idx

def normalize(X):
    # ensure X is on same device/dtype as mu/std
    return (X - mu) / std


upper_limit, lower_limit = 1.0, 0.0


def clamp(X, lower=lower_limit, upper=upper_limit):
    """Stable clamp that handles scalars or tensors; returns same dtype/device as X."""
    # convert bounds to tensors on X.device and X.dtype
    if not torch.is_tensor(lower):
        lower = torch.tensor(lower, dtype=X.dtype, device=X.device)
    else:
        lower = lower.to(device=X.device, dtype=X.dtype)
    if not torch.is_tensor(upper):
        upper = torch.tensor(upper, dtype=X.dtype, device=X.device)
    else:
        upper = upper.to(device=X.device, dtype=X.dtype)
    return torch.max(torch.min(X, upper), lower)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0], device=device, dtype=torch.float32)
    max_delta = torch.zeros_like(X, device=X.device)
    for _ in range(restarts):
        delta = torch.zeros_like(X, device=X.device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--train-fraction', type=float, default=1.0)
    parser.add_argument('--resume-from', default='', type=str)
    parser.add_argument('--autosave-every', type=int, default=500)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--attack-iters-test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    parser.add_argument('--awp-gamma', default=0.01, type=float)
    parser.add_argument('--awp-warmup', default=0, type=int)
    return parser.parse_args()

def save_checkpoint(state, fname, logger=None):
    """
    Atomic checkpoint save.
    - state: dict to save (torch.save-able)
    - fname: target path (string or Path)
    - logger: optional logger to log success
    """
    fname = str(fname)
    d = os.path.dirname(fname)
    if d:
        os.makedirs(d, exist_ok=True)

    tmp = fname + '.tmp'
    torch.save(state, tmp)
    Path(tmp).replace(fname)
    if logger is not None:
        logger.info(f"Saved checkpoint: {fname}")

def main():
    args = get_args()
    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            dataset = torch.load("cifar10_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        dataset = cifar10(args.data_dir)
    train_set_full = list(zip(transpose(pad(dataset['train']['data'], 4)/255.), dataset['train']['labels']))
    if ckpt is not None and 'train_subset_indices' in ckpt and ckpt['train_subset_indices'] is not None:
        train_subset_indices = np.array(ckpt['train_subset_indices'], dtype=int)
        train_set = [train_set_full[i] for i in train_subset_indices]
        logger.info(f"Restored train subset of size {len(train_set)} from checkpoint")
    else:
        # fallback: deterministic subsample from args.seed if requested
        if args.train_fraction < 1.0:
            rng = np.random.RandomState(args.seed)
            train_subset_indices = rng.permutation(len(train_set_full))[:int(len(train_set_full) * args.train_fraction)]
            train_set = [train_set_full[i] for i in train_subset_indices]
            logger.info(f"Using {len(train_set)} / {len(train_set_full)} training samples (seed={args.seed})")
        else:
            train_subset_indices = None
            train_set = train_set_full
    # ----------------------------------
    
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(
        train_set_x,
        args.batch_size,
        shuffle=True,
        set_random_choices=True,
        num_workers=2
    )

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    if args.model == 'PreActResNet18':
        model = PreActResNet18()
        proxy = PreActResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
        proxy = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

        # --- create model & optimizers first (so we can load into them) ---
        if args.model == 'PreActResNet18':
            model = PreActResNet18()
            proxy = PreActResNet18()
        elif args.model == 'WideResNet':
            model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
            proxy = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
        else:
            raise ValueError("Unknown model")
    
        # wrap and move to device
        model = nn.DataParallel(model).to(device)
        proxy = nn.DataParallel(proxy).to(device)
    
        # set up optimizers (so checkpoint can restore them)
        if args.l2:
            decay, no_decay = [], []
            for name,param in model.named_parameters():
                if 'bn' not in name and 'bias' not in name:
                    decay.append(param)
                else:
                    no_decay.append(param)
            params = [{'params':decay, 'weight_decay':args.l2},
                      {'params':no_decay, 'weight_decay': 0 }]
        else:
            params = model.parameters()
    
        opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
        proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
    
        # bookkeeping defaults
        start_epoch = 0
        resume_batch_idx = 0
        best_test_robust_acc = 0.0
        best_val_robust_acc = 0.0
    
        # --- load checkpoint if present (explicit resume-from or model_latest) ---
        ckpt_path = None
        if args.resume_from:
            ckpt_path = args.resume_from
        else:
            candidate = os.path.join(args.fname, 'model_latest.pth')
            if os.path.exists(candidate):
                ckpt_path = candidate
    
        if ckpt_path is not None:
            logger.info(f"Attempting to load checkpoint: {ckpt_path}")
            try:
                # load to CPU first (safe) and then restore into model/optimizers
                loaded = safe_torch_load(ckpt_path, map_location='cpu')
                # load model (permissive)
                if 'model_state' in loaded:
                    try:
                        model.load_state_dict(loaded['model_state'])
                        logger.info("Model weights loaded from checkpoint.")
                    except Exception as e:
                        try:
                            model.load_state_dict(loaded['model_state'], strict=False)
                            logger.warning(f"Model loaded with strict=False: {e}")
                        except Exception as e2:
                            logger.warning(f"Model load failed: {e2}")
    
                # load optimizer states if present
                if 'opt_state' in loaded:
                    try:
                        opt.load_state_dict(loaded['opt_state'])
                        logger.info("Optimizer state restored.")
                    except Exception as e:
                        logger.warning(f"Could not restore optimizer state: {e}")
    
                if 'proxy_state' in loaded and loaded['proxy_state'] is not None:
                    try:
                        proxy.load_state_dict(loaded['proxy_state'])
                    except Exception as e:
                        logger.warning(f"Could not restore proxy state: {e}")
                if 'proxy_opt_state' in loaded and loaded['proxy_opt_state'] is not None:
                    try:
                        proxy_opt.load_state_dict(loaded['proxy_opt_state'])
                    except Exception as e:
                        logger.warning(f"Could not restore proxy optimizer state: {e}")
    
                # restore RNGs (best-effort)
                try:
                    if 'rng_numpy' in loaded: np.random.set_state(loaded['rng_numpy'])
                    if 'rng_python' in loaded: pyrandom.setstate(loaded['rng_python'])
                    if 'rng_torch' in loaded:
                        rt = loaded['rng_torch']
                        if isinstance(rt, torch.Tensor):
                            torch.set_rng_state(rt)
                        else:
                            # attempt conversion for older pickles
                            torch.set_rng_state(torch.tensor(rt, dtype=torch.uint8))
                    if torch.cuda.is_available() and 'rng_cuda_all' in loaded:
                        cuda_states = []
                        for s in loaded['rng_cuda_all']:
                            cuda_states.append(torch.tensor(s, dtype=torch.uint8) if not isinstance(s, torch.Tensor) else s)
                        torch.cuda.set_rng_state_all(cuda_states)
                except Exception as e:
                    logger.warning(f"RNG restore failed: {e}")
    
                # bookkeeping
                start_epoch = int(loaded.get('epoch', 0))
                resume_batch_idx = int(loaded.get('batch_idx', 0))
                best_test_robust_acc = loaded.get('best_test_robust_acc', best_test_robust_acc)
                best_val_robust_acc = loaded.get('best_val_robust_acc', best_val_robust_acc)
                # train_subset_indices if present
                if 'train_subset_indices' in loaded:
                    train_subset_indices = loaded.get('train_subset_indices', train_subset_indices)
                logger.info(f"Resuming from epoch={start_epoch}, batch_idx={resume_batch_idx}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")
        else:
            logger.info("No checkpoint found; training from scratch.")


    awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()
    
    if ckpt is not None:
        # model weights
        if 'model_state' in ckpt:
            try:
                model.load_state_dict(ckpt['model_state'])
                logger.info("Loaded model weights from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not fully load model state from checkpoint: {e}")

        # optimizer state
        if 'opt_state' in ckpt:
            try:
                opt.load_state_dict(ckpt['opt_state'])
                logger.info("Loaded optimizer state from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not fully load optimizer state from checkpoint: {e}")

        # optionally restore other bookkeeping
        best_test_robust_acc = ckpt.get('best_test_robust_acc', -1.0)
        best_val_robust_acc = ckpt.get('best_val_robust_acc', -1.0)

        # RNG states (optional but recommended for reproducibility)
        import numpy as _np
        import types
        
        def _ensure_byte_tensor(x):
            """Return a torch.ByteTensor (dtype=uint8) on CPU suitable for set_rng_state.
               Raises ValueError if conversion is impossible.
            """
            # If it's a torch tensor, move to CPU and convert dtype if needed.
            if isinstance(x, torch.Tensor):
                t = x.cpu()
                if t.dtype != torch.uint8:
                    # Convert numeric representation to uint8 preserving raw bytes
                    try:
                        t = t.to(dtype=torch.uint8)
                    except Exception:
                        # fallback: coerce via numpy
                        t = torch.from_numpy(t.numpy().astype(_np.uint8))
                return t.contiguous()
        
            # Numpy array -> uint8 tensor on CPU
            if isinstance(x, _np.ndarray):
                return torch.from_numpy(x.astype(_np.uint8)).contiguous()
        
            # bytes/bytearray -> list of ints -> tensor
            if isinstance(x, (bytes, bytearray)):
                return torch.tensor(list(x), dtype=torch.uint8).contiguous()
        
            # list / tuple / generator -> tensor
            if isinstance(x, (list, tuple, types.GeneratorType)):
                return torch.tensor(list(x), dtype=torch.uint8).contiguous()
        
            # fallback attempt
            try:
                return torch.tensor(list(x), dtype=torch.uint8).contiguous()
            except Exception as e:
                raise ValueError(f"Cannot convert RNG state of type {type(x)} to torch.uint8: {e}")
        
        # restore RNGs if present (robust conversions)
        if 'rng_numpy' in ckpt:
            np.random.set_state(ckpt['rng_numpy'])
        if 'rng_python' in ckpt:
            pyrandom.setstate(ckpt['rng_python'])
        
        if 'rng_torch' in ckpt:
            try:
                rng_cpu = _ensure_byte_tensor(ckpt['rng_torch'])
                torch.set_rng_state(rng_cpu)
                logger.info("Restored CPU RNG state from checkpoint.")
            except Exception as e:
                logger.warning(f"Failed to set CPU RNG state from checkpoint: {e}")
                logger.debug(f"rng_torch raw type: {type(ckpt.get('rng_torch'))}")
        
        if torch.cuda.is_available() and 'rng_cuda_all' in ckpt:
            try:
                cuda_raw = ckpt['rng_cuda_all']
                # accept list-like of states; convert each to ByteTensor and ensure on CPU (set_rng_state_all expects cpu tensors)
                cuda_converted = []
                for s in cuda_raw:
                    t = _ensure_byte_tensor(s)
                    cuda_converted.append(t)
                torch.cuda.set_rng_state_all(cuda_converted)
                logger.info("Restored CUDA RNG states from checkpoint.")
            except Exception as e:
                logger.warning(f"Failed to set CUDA RNG states from checkpoint: {e}")
                logger.debug(f"rng_cuda_all raw type: {type(ckpt.get('rng_cuda_all'))}")
        # ---------------------------------------------------------------------------




        # determine start epoch if present in checkpoint
        if 'epoch' in ckpt:
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            logger.info(f"Resuming training from epoch {start_epoch}")

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32, device=device)
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32, device=device)
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    elif args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, 0.4 * args.epochs, args.epochs], [0, args.lr_max, 0])[0]

    best_test_robust_acc = 0
    best_val_robust_acc = 0
    start_epoch = 0
# prefer explicit resume-from path if provided
    if args.resume_from:
        ckpt_path = args.resume_from
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = safe_torch_load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        if 'opt_state' in ckpt:
            try:
                opt.load_state_dict(ckpt['opt_state'])
            except Exception as e:
                logger.warning(f"Could not load optimizer state cleanly: {e}")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_test_robust_acc = ckpt.get('best_test_robust_acc', best_test_robust_acc)
        best_val_robust_acc = ckpt.get('best_val_robust_acc', best_val_robust_acc)
    elif args.resume:
        start_epoch = args.resume
        model.load_state_dict(safe_torch_load(os.path.join(args.fname, f'model_{start_epoch-1}.pth'), map_location=device))
        opt.load_state_dict(safe_torch_load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth'), map_location=device))
        logger.info(f'Resuming at epoch {start_epoch}')
        if os.path.exists(os.path.join(args.fname, f'model_best.pth')):
            best_test_robust_acc = safe_torch_load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
        if args.val and os.path.exists(os.path.join(args.fname, f'model_val.pth')):
            best_val_robust_acc = safe_torch_load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0


    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    try:
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_robust_loss = 0
            train_robust_acc = 0
            train_n = 0
            # Build iterator and (if resuming) advance to saved batch index
            train_iter = iter(train_batches)
            # Only advance if we're resuming at this epoch
            if start_epoch is not None and start_epoch == epoch and resume_batch_idx > 0:
                logger.info(f"Advancing train iterator to batch {resume_batch_idx} for mid-epoch resume")
                # advance the iterator resume_batch_idx times
                for _skip in range(resume_batch_idx):
                    try:
                        next(train_iter)
                    except StopIteration:
                        break
                # after advancing, set i to resume_batch_idx so logging matches
                i_start = resume_batch_idx
            else:
                i_start = 0
            
            for i_offset, batch in enumerate(train_iter, start=i_start):
                i = i_offset  # batch index relative to epoch
                if args.eval:
                    break
                X, y = batch['input'], batch['target']
                if args.mixup:
                    X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                    X, y_a, y_b = X.to(device), y_a.to(device), y_b.to(device)
                lr = lr_schedule(epoch + (i + 1) / len(train_batches))
                for pg in opt.param_groups:
                    pg['lr'] = lr
    
                if args.attack == 'pgd':
                    # Random initialization
                    if args.mixup:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                    else:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                    delta = delta.detach()
                elif args.attack == 'fgsm':
                    delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
                # Standard training
                elif args.attack == 'none':
                    delta = torch.zeros_like(X)
                X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
    
                model.train()
                # calculate adversarial weight perturbation and perturb it
                if epoch >= args.awp_warmup:
                    # not compatible to mixup currently.
                    assert (not args.mixup)
                    awp = awp_adversary.calc_awp(inputs_adv=X_adv,
                                                 targets=y)
                    awp_adversary.perturb(awp)
    
                robust_output = model(X_adv)
                if args.mixup:
                    robust_loss = mixup_criterion(criterion, robust_output, y_a, y_b, lam)
                else:
                    robust_loss = criterion(robust_output, y)
    
                if args.l1:
                    for name,param in model.named_parameters():
                        if 'bn' not in name and 'bias' not in name:
                            robust_loss += args.l1*param.abs().sum()
    
                opt.zero_grad()
                robust_loss.backward()
                opt.step()
                if args.autosave_every > 0 and ((i + 1) % args.autosave_every == 0):
                    latest_path = os.path.join(args.fname, 'model_latest.pth')
                    # save batch index as next batch to execute (i+1)
                    save_full_checkpoint(
                        latest_path,
                        model=model,
                        opt=opt,
                        proxy=proxy,
                        proxy_opt=proxy_opt,
                        awp=awp_adversary if 'awp_adversary' in locals() else None,
                        scaler=None,
                        scheduler=None,
                        epoch=epoch,
                        batch_idx=(i + 1),
                        best_test=best_test_robust_acc,
                        best_val=best_val_robust_acc,
                        train_subset_indices=train_subset_indices,
                        logger=logger
                    )

                if epoch >= args.awp_warmup:
                    awp_adversary.restore(awp)
    
                output = model(normalize(X))
                if args.mixup:
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                else:
                    loss = criterion(output, y)
    
                train_robust_loss += robust_loss.item() * y.size(0)
                train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
    
            train_time = time.time()
    
            model.eval()
            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0
    
            with torch.no_grad():
                for i, batch in enumerate(test_batches):
                    X, y = batch['input'], batch['target']
        
                    # Random initialization
                    if args.attack == 'none':
                        delta = torch.zeros_like(X)
                    else:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
                    delta = delta.detach()
        
                    robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    robust_loss = criterion(robust_output, y)
        
                    output = model(normalize(X))
                    loss = criterion(output, y)
        
                    test_robust_loss += robust_loss.item() * y.size(0)
                    test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                    test_loss += loss.item() * y.size(0)
                    test_acc += (output.max(1)[1] == y).sum().item()
                    test_n += y.size(0)
        
                test_time = time.time()
    
            if args.val:
                val_loss = 0
                val_acc = 0
                val_robust_loss = 0
                val_robust_acc = 0
                val_n = 0
                with torch.no_grad():
                    for i, batch in enumerate(val_batches):
                        X, y = batch['input'], batch['target']
        
                        # Random initialization
                        if args.attack == 'none':
                            delta = torch.zeros_like(X)
                        else:
                            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
                        delta = delta.detach()
        
                        robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                        robust_loss = criterion(robust_output, y)
        
                        output = model(normalize(X))
                        loss = criterion(output, y)
        
                        val_robust_loss += robust_loss.item() * y.size(0)
                        val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                        val_loss += loss.item() * y.size(0)
                        val_acc += (output.max(1)[1] == y).sum().item()
                        val_n += y.size(0)
    
            if not args.eval:
                logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                    epoch, train_time - start_time, test_time - train_time, lr,
                    train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                    test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
    
                if args.val:
                    logger.info('validation %.4f \t %.4f \t %.4f \t %.4f',
                        val_loss/val_n, val_acc/val_n, val_robust_loss/val_n, val_robust_acc/val_n)
    
                    if val_robust_acc/val_n > best_val_robust_acc:
                        torch.save({
                                'state_dict':model.state_dict(),
                                'test_robust_acc':test_robust_acc/test_n,
                                'test_robust_loss':test_robust_loss/test_n,
                                'test_loss':test_loss/test_n,
                                'test_acc':test_acc/test_n,
                                'val_robust_acc':val_robust_acc/val_n,
                                'val_robust_loss':val_robust_loss/val_n,
                                'val_loss':val_loss/val_n,
                                'val_acc':val_acc/val_n,
                            }, os.path.join(args.fname, f'model_val.pth'))
                        best_val_robust_acc = val_robust_acc/val_n
    
                # save checkpoint
                if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                    # save regular per-epoch files (for easy inspection)
                    torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
                    torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))
                    # save a full checkpoint including metadata (batch_idx=0 for next epoch)
                    epoch_ckpt = os.path.join(args.fname, f'model_epoch_{epoch}.pth')
                    save_full_checkpoint(
                        epoch_ckpt, model, opt,
                        proxy=proxy, proxy_opt=proxy_opt,
                        awp=awp_adversary if 'awp_adversary' in locals() else None,
                        scaler=None, scheduler=None,
                        epoch=epoch + 1, batch_idx=0,
                        best_test=best_test_robust_acc, best_val=best_val_robust_acc,
                        train_subset_indices=train_subset_indices,
                        logger=logger)
                    # also update latest
                    latest_path = os.path.join(args.fname, 'model_latest.pth')
                    save_full_checkpoint(latest_path, model, opt,
                                         proxy=proxy, proxy_opt=proxy_opt,
                                         awp=awp_adversary if 'awp_adversary' in locals() else None,
                                         scaler=None, scheduler=None,
                                         epoch=epoch + 1, batch_idx=0,
                                         best_test=best_test_robust_acc, best_val=best_val_robust_acc,
                                         train_subset_indices=train_subset_indices,
                                         logger=logger)

    
                # save best
                if test_robust_acc/test_n > best_test_robust_acc:
                    best_path = os.path.join(args.fname, f'model_best.pth')
                    save_full_checkpoint(
                        best_path, model, opt,
                        proxy=proxy, proxy_opt=proxy_opt,
                        awp=awp_adversary if 'awp_adversary' in locals() else None,
                        scaler=None, scheduler=None,
                        epoch=epoch + 1, batch_idx=0,
                        best_test=test_robust_acc/test_n, best_val=best_val_robust_acc,
                        train_subset_indices=train_subset_indices,
                        logger=logger)
                    best_test_robust_acc = test_robust_acc/test_n
            else:
                logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                    epoch, train_time - start_time, test_time - train_time, -1,
                    -1, -1, -1, -1,
                    test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
                return
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught — saving interrupt checkpoint...")
        state_path = os.path.join(args.fname, 'model_interrupt.pth')
        save_full_checkpoint(
            state_path, model, opt,
            proxy=proxy, proxy_opt=proxy_opt,
            awp=awp_adversary if 'awp_adversary' in locals() else None,
            scaler=None, scheduler=None,
            epoch=epoch, batch_idx=i,
            best_test=best_test_robust_acc, best_val=best_val_robust_acc,
            train_subset_indices=train_subset_indices,
            logger=logger)
        raise

    except Exception as e:
        logger.exception("Exception during training — saving crash checkpoint...")
        state_path = os.path.join(args.fname, 'model_crash.pth')
        # include exception string in extra
        save_full_checkpoint(
            state_path, model, opt,
            proxy=proxy, proxy_opt=proxy_opt,
            awp=awp_adversary if 'awp_adversary' in locals() else None,
            scaler=None, scheduler=None,
            epoch=epoch, batch_idx=i,
            best_test=best_test_robust_acc, best_val=best_val_robust_acc,
            train_subset_indices=train_subset_indices,
            logger=logger,
            extra={'exception': str(e)}
        )
        raise

if __name__ == "__main__":
    main()
