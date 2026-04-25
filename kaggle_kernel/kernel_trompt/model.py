"""Trompt model + training loop via pytorch_frame.

Trompt (Chen et al. 2023): prompt-based column attention for tabular.
Key params (per the published kernel):
  channels=128, num_prompts=128, num_layers=3
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_frame import stype
from torch_frame.data import DataLoader, Dataset
from torch_frame.nn.models import Trompt


def make_dataset(df, cats, nums, target=None):
    col_to_stype = {c: stype.categorical for c in cats}
    col_to_stype.update({c: stype.numerical for c in nums})
    if target is not None:
        col_to_stype[target] = stype.categorical
    ds = Dataset(df=df, col_to_stype=col_to_stype, target_col=target)
    ds.materialize()
    return ds


def build_model(train_ds, num_classes: int, device,
                channels: int = 128, num_prompts: int = 128,
                num_layers: int = 3) -> Trompt:
    model = Trompt(
        channels=channels,
        out_channels=num_classes,
        num_prompts=num_prompts,
        num_layers=num_layers,
        col_stats=train_ds.col_stats,
        col_names_dict=train_ds.tensor_frame.col_names_dict,
    ).to(device)
    return model


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for tf in loader:
        tf = tf.to(device)
        optimizer.zero_grad()
        out = model(tf).mean(dim=1)  # average over prompts
        y = tf.y.long()
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def predict_proba(model, loader, device) -> torch.Tensor:
    model.eval()
    probs = []
    for tf in loader:
        tf = tf.to(device)
        out = model(tf).mean(dim=1)
        probs.append(F.softmax(out, dim=1).cpu())
    return torch.cat(probs, dim=0)


def fit_one_fold(train_ds, val_ds, test_ds, device,
                 n_epochs: int, batch_size: int = 512,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 num_classes: int = 3, channels: int = 128,
                 num_prompts: int = 128, num_layers: int = 3):
    model = build_model(train_ds, num_classes, device, channels,
                        num_prompts, num_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=n_epochs)
    tl = DataLoader(train_ds.tensor_frame, batch_size=batch_size,
                    shuffle=True)
    vl = DataLoader(val_ds.tensor_frame, batch_size=batch_size)
    el = DataLoader(test_ds.tensor_frame, batch_size=batch_size)
    for ep in range(n_epochs):
        loss = train_epoch(model, tl, opt, device)
        sched.step()
        if ep % 5 == 0 or ep == n_epochs - 1:
            print(f"      epoch {ep+1}/{n_epochs} train_loss={loss:.4f}",
                  flush=True)
    p_val = predict_proba(model, vl, device).numpy()
    p_test = predict_proba(model, el, device).numpy()
    return p_val, p_test
