import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
CHECKPOINT_PATH = "/content/drive/MyDrive/TDL/AWP/AWP_10%/model_best.pth"   # change if your checkpoint is elsewhere
FIXED_BATCH_PATH = "/mnt/data/fixed_batch.pt"
SAVE_PATH = "/content/drive/MyDrive/TDL/AWP/loss_landscape_resnet18.png"

NUM_POINTS = 21         # grid resolution per axis (reduce to 11 for speed)
ALPHA_RANGE = 0.05      # ± fraction of weight norm for directions
PGD_STEPS = 10
PGD_STEP_SIZE = 2/255.0
ATTACK_EPS = 8/255.0
BATCH_SIZE = 128        # size of fixed minibatch to use
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------

print("Device:", DEVICE)
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

# ---------------- Import ResNet18 from repo ----------------
# tries a few common names to be robust
resnet_candidates = [
    ("resnet", "ResNet18"),
    ("resnet", "ResNet"),       # common alt
    ("preactresnet", "PreActResNet18"),
]
ModelClass = None
for mod_name, cls_name in resnet_candidates:
    try:
        # import module from local repo
        module = __import__(mod_name)
        ModelClass = getattr(module, cls_name)
        print(f"Imported {cls_name} from {mod_name}.py")
        break
    except Exception:
        ModelClass = None

if ModelClass is None:
    # try importing any class named ResNet18 in resnet module
    try:
        import resnet as _resmod
        ModelClass = getattr(_resmod, "ResNet18", None) or getattr(_resmod, "ResNet", None)
        if ModelClass:
            print("Imported ResNet class from resnet.py")
    except Exception:
        ModelClass = None

if ModelClass is None:
    raise ImportError("Could not find ResNet class in your repo. Ensure resnet.py exists and defines ResNet18 or ResNet.")

# ---------------- Build model and load checkpoint ----------------
# (ensure CHECKPOINT_PATH and DEVICE are set above)
model = ModelClass(num_classes=10).to(DEVICE)  # change num_classes if needed

# Robust checkpoint loading block
import torch
from pprint import pprint

def extract_state_dict(ckpt):
    candidates = [
        "state_dict",
        "model_state",
        "model_state_dict",
        "model",
        "model_state_dict_ema",
        "network",
    ]
    for c in candidates:
        if isinstance(ckpt, dict) and c in ckpt:
            candidate = ckpt[c]
            if isinstance(candidate, dict) and "state_dict" in candidate:
                return candidate["state_dict"]
            return candidate
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if isinstance(v, dict):
                sample_vals = list(v.values())[:4]
                if sample_vals and all(hasattr(x, "ndim") or isinstance(x, torch.Tensor) for x in sample_vals):
                    return v
    return None

# load checkpoint allowing non-weights (trusted local file)
try:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
except TypeError:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

print("Top-level checkpoint type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Top-level keys:")
    pprint(list(ckpt.keys())[:50])

state = extract_state_dict(ckpt)
if state is None:
    raise RuntimeError("Could not find a state_dict-like object in the checkpoint. Inspect printed keys above.")

if isinstance(state, dict):
    print("Found candidate state-dict with keys (sample):", list(state.keys())[:10])

def strip_module_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        if new_key.startswith("model_state."):
            new_key = new_key[len("model_state."):]
        new_sd[new_key] = v
    return new_sd

if not isinstance(state, dict) or not all(isinstance(k, str) for k in state.keys()):
    raise RuntimeError("Extracted state is not a dict of parameter tensors. Inspect checkpoint structure manually.")

# Try to load
loaded = False
try:
    model.load_state_dict(state)
    print("Loaded state_dict with strict=True (direct).")
    loaded = True
except Exception as e_strict:
    print("Strict load failed, attempting to strip prefixes and retry. Error:")
    print(e_strict)
    stripped = strip_module_prefix(state)
    try:
        model.load_state_dict(stripped)
        print("Loaded state_dict after stripping prefixes (strict=True).")
        loaded = True
    except Exception as e2:
        print("Still failed to load with strict=True after stripping prefixes. Error:")
        print(e2)
        load_res = model.load_state_dict(stripped, strict=False)
        print("Loaded with strict=False. Missing keys (will be randomly initialized):")
        pprint(load_res.missing_keys)
        print("Unexpected keys in checkpoint (ignored):")
        pprint(load_res.unexpected_keys)
        if len(load_res.missing_keys) > 0:
            print("\nWARNING: many missing keys — check that the ResNet variant and num_classes match training.")
        loaded = True  # we still set True because strict=False did load params that exist

if loaded:
    model.eval()
    print("Model ready. Moving on to landscape computation.")

# ---------------- Fixed minibatch (create if missing) ----------------
if not os.path.exists(FIXED_BATCH_PATH):
    print(f"No fixed batch found at {FIXED_BATCH_PATH}. Attempting to create from torchvision CIFAR-10.")
    try:
        import torchvision
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor()])
        ds = torchvision.datasets.CIFAR10(root="/mnt/data", train=False, download=True, transform=transform)
        subset_idx = list(range(min(BATCH_SIZE, len(ds))))
        subset = Subset(ds, subset_idx)
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=0)
        inputs, targets = next(iter(loader))
        torch.save({"inputs": inputs, "targets": targets}, FIXED_BATCH_PATH)
        print("Saved fixed batch to", FIXED_BATCH_PATH)
    except Exception as e:
        print("Failed to create fixed batch automatically. Exception:", e)
        raise RuntimeError("Create a fixed_batch.pt at FIXED_BATCH_PATH manually and rerun.")

fb = torch.load(FIXED_BATCH_PATH, map_location=DEVICE, weights_only=False)
inputs = fb["inputs"].to(DEVICE)
targets = fb["targets"].to(DEVICE)
print("Using fixed batch:", inputs.shape)

# ---------------- Robust loss (PGD inner maximization) ----------------
loss_fn = nn.CrossEntropyLoss(reduction="mean")

def pgd_robust_loss(model, x, y, eps=ATTACK_EPS, steps=PGD_STEPS, step_size=PGD_STEP_SIZE, device=DEVICE):
    model.eval()
    x0 = x.detach()
    delta = torch.zeros_like(x0).uniform_(-eps, eps).to(device)
    delta.requires_grad_(True)
    for _ in range(steps):
        logits = model(x0 + delta)
        loss = loss_fn(logits, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = (delta + step_size * torch.sign(grad)).clamp(-eps, eps)
        delta.grad.zero_()
    with torch.no_grad():
        logits = model(x0 + delta)
        final_loss = loss_fn(logits, y).item()
    return final_loss

# ---------------- Parameter vector helpers ----------------
param_list = [p for p in model.parameters() if p.requires_grad]
param_sizes = [p.numel() for p in param_list]
total_params = sum(param_sizes)
print("Total trainable params:", total_params)

def get_param_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad]).cpu().numpy()

def set_param_vector(model, vec):
    vec = vec.astype(np.float32)
    ptr = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        new_val = torch.from_numpy(vec[ptr:ptr+n].reshape(p.shape)).to(DEVICE)
        p.data.copy_(new_val)
        ptr += n

base_vec = get_param_vector(model)
base_norm = np.linalg.norm(base_vec)
print("Base param norm:", base_norm)

# random directions
rng = np.random.RandomState(0)
d1 = rng.randn(total_params).astype(np.float32)
d2 = rng.randn(total_params).astype(np.float32)
d1 = d1 / np.linalg.norm(d1) * base_norm
d2 = d2 / np.linalg.norm(d2) * base_norm

alphas = np.linspace(-ALPHA_RANGE, ALPHA_RANGE, NUM_POINTS)
betas = np.linspace(-ALPHA_RANGE, ALPHA_RANGE, NUM_POINTS)
Z = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.float32)

print(f"Evaluating {NUM_POINTS}x{NUM_POINTS} grid (this may take time).")
for i, a in enumerate(alphas):
    for j, b in enumerate(betas):
        pert = base_vec + a * d1 + b * d2
        set_param_vector(model, pert)
        val = pgd_robust_loss(model, inputs, targets)
        Z[j, i] = val
    print(f"Column {i+1}/{len(alphas)} done.")

# restore weights
set_param_vector(model, base_vec)

# ---------------- Plot ----------------
X, Y = np.meshgrid(alphas, betas)
plt.figure(figsize=(6,5))
cf = plt.contourf(X, Y, Z, levels=50)
plt.colorbar(cf)
plt.xlabel("alpha (direction 1 scale)")
plt.ylabel("beta (direction 2 scale)")
plt.title("Robust loss landscape (ResNet18)")
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=200)
plt.show()
print("Saved landscape to:", SAVE_PATH)
