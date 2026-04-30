"""L2 — SupCon embedding (verbatim port of p3_embed_propagate.Embedder).

Small MLP backbone with two heads:
  - classifier (cross-entropy)
  - projection (supervised contrastive)
joint loss = CE + 0.5 * SupCon. 32-d L2-normalised projection.
"""
from __future__ import annotations

import time
import numpy as np


def _select_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedder:
    def __init__(self, in_dim: int, embed_dim: int = 32, device=None):
        import torch
        import torch.nn as nn
        self.torch = torch; self.nn = nn; self.device = device
        hidden = 256
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
        ).to(device)
        self.classifier = nn.Linear(hidden, 3).to(device)
        self.projection = nn.Sequential(
            nn.Linear(hidden, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(device)
        self.embed_dim = embed_dim

    def _supcon_loss(self, z, y, temperature=0.5):
        z = self.nn.functional.normalize(z, dim=1)
        sim = z @ z.t() / temperature
        sim.fill_diagonal_(-1e9)
        mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        mask.fill_diagonal_(0)
        exp_sim = sim.exp()
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12
        log_prob = sim - denom.log()
        has_pos = (mask.sum(dim=1) > 0)
        if has_pos.sum() == 0:
            return self.torch.tensor(0.0, device=z.device)
        pos_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return -pos_log_prob[has_pos].mean()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, bs: int):
        torch = self.torch
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X)).float(),
            torch.from_numpy(np.ascontiguousarray(y)).long())
        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(
            list(self.backbone.parameters())
            + list(self.classifier.parameters())
            + list(self.projection.parameters()),
            lr=1e-3, weight_decay=1e-4)
        self.backbone.train(); self.classifier.train(); self.projection.train()
        for ep in range(epochs):
            t0 = time.time(); losses = []
            for xb, yb in loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                h = self.backbone(xb)
                logits = self.classifier(h)
                z = self.projection(h)
                loss_ce = self.nn.functional.cross_entropy(logits, yb)
                loss_sup = self._supcon_loss(z, yb)
                loss = loss_ce + 0.5 * loss_sup
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            log(f"    epoch {ep+1}/{epochs} loss={np.mean(losses):.4f} "
                f"dt={time.time()-t0:.1f}s")

    def transform(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.backbone.eval(); self.projection.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 8192):
                xb = torch.from_numpy(np.ascontiguousarray(X[i:i+8192])).float().to(self.device)
                z = self.projection(self.backbone(xb))
                z = self.nn.functional.normalize(z, dim=1).cpu().numpy()
                out.append(z)
        return np.concatenate(out, axis=0).astype(np.float32)
