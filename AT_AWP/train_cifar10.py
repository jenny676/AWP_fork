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

    ckpt = None
    if args.resume_from:
        logger.info(f"Loading checkpoint for resume-from: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)


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

    model = nn.DataParallel(model).to(device)
    proxy = nn.DataParallel(proxy).to(device)

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
        if 'rng_numpy' in ckpt:
            np.random.set_state(ckpt['rng_numpy'])
        if 'rng_python' in ckpt:
            pyrandom.setstate(ckpt['rng_python'])
        if 'rng_torch' in ckpt:
            torch.set_rng_state(ckpt['rng_torch'])
        if torch.cuda.is_available() and 'rng_cuda_all' in ckpt:
            torch.cuda.set_rng_state_all(ckpt['rng_cuda_all'])

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
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        if 'opt_state' in ckpt:
            try:
                opt.load_state_dict(ckpt['opt_state'])
            except Exception as e:
                logger.warning(f"Could not load optimizer state cleanly: {e}")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_test_robust_acc = ckpt.get('best_test_robust_acc', best_test_robust_acc)
        best_val_robust_acc = ckpt.get('best_val_robust_acc', best_val_robust_acc)
        # restore RNGs if present (optional)
        if 'rng_numpy' in ckpt:
            np.random.set_state(ckpt['rng_numpy'])
        if 'rng_python' in ckpt:
            pyrandom.setstate(ckpt['rng_python'])
        if 'rng_torch' in ckpt:
            torch.set_rng_state(ckpt['rng_torch'])
        if torch.cuda.is_available() and 'rng_cuda_all' in ckpt:
            torch.cuda.set_rng_state_all(ckpt['rng_cuda_all'])
    elif args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth'), map_location=device))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth'), map_location=device))
        logger.info(f'Resuming at epoch {start_epoch}')
        if os.path.exists(os.path.join(args.fname, f'model_best.pth')):
            best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
        if args.val and os.path.exists(os.path.join(args.fname, f'model_val.pth')):
            best_val_robust_acc = torch.load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
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
            for i, batch in enumerate(train_batches):
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
                    state = {
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'opt_state': opt.state_dict(),
                        'best_test_robust_acc': best_test_robust_acc,
                        'best_val_robust_acc': best_val_robust_acc,
                        'rng_numpy': np.random.get_state(),
                        'rng_python': pyrandom.getstate(),
                        'rng_torch': torch.get_rng_state(),
                        'train_subset_indices': None if train_subset_indices is None else train_subset_indices.tolist(),
                    }
                    if torch.cuda.is_available():
                        state['rng_cuda_all'] = torch.cuda.get_rng_state_all()
                    latest_path = os.path.join(args.fname, 'model_latest.pth')
                    save_checkpoint(state, latest_path, logger)
    
    
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
                    torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
                    torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))
    
                # save best
                if test_robust_acc/test_n > best_test_robust_acc:
                    torch.save({
                            'state_dict':model.state_dict(),
                            'test_robust_acc':test_robust_acc/test_n,
                            'test_robust_loss':test_robust_loss/test_n,
                            'test_loss':test_loss/test_n,
                            'test_acc':test_acc/test_n,
                        }, os.path.join(args.fname, f'model_best.pth'))
                    best_test_robust_acc = test_robust_acc/test_n
            else:
                logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                    epoch, train_time - start_time, test_time - train_time, -1,
                    -1, -1, -1, -1,
                    test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
                return
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught — saving interrupt checkpoint...")
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'best_test_robust_acc': best_test_robust_acc,
            'best_val_robust_acc': best_val_robust_acc,
            'rng_numpy': np.random.get_state(),
            'rng_python': pyrandom.getstate(),
            'rng_torch': torch.get_rng_state(),
            'train_subset_indices': None if train_subset_indices is None else train_subset_indices.tolist(),
        }
        if torch.cuda.is_available():
            state['rng_cuda_all'] = torch.cuda.get_rng_state_all()
        save_checkpoint(state, os.path.join(args.fname, 'model_interrupt.pth'), logger)
        raise
    except Exception as e:
        logger.exception("Exception during training — saving crash checkpoint...")
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'best_test_robust_acc': best_test_robust_acc,
            'best_val_robust_acc': best_val_robust_acc,
            'rng_numpy': np.random.get_state(),
            'rng_python': pyrandom.getstate(),
            'rng_torch': torch.get_rng_state(),
            'exception': str(e),
            'train_subset_indices': None if train_subset_indices is None else train_subset_indices.tolist(),
        }
        if torch.cuda.is_available():
            state['rng_cuda_all'] = torch.cuda.get_rng_state_all()
        save_checkpoint(state, os.path.join(args.fname, 'model_crash.pth'), logger)
        raise


if __name__ == "__main__":
    main()
