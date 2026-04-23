"""Option 1 — Two-stage DGP reproduction: SDV TVAE features + MLP labeler.

Hypothesis (per CLAUDE.md 2026-04-21 DGP-residuals EDA + brief.md:74):
the host generated the 630k synthetic dataset by

  (A) Training a deep tabular generator (we model SDV TVAE) on the 10k
      original *features* (label dropped). Sample N_GEN feature rows.
  (B) Training a deep classifier (small MLP) on the 10k original with
      rule-perfect labels, stopped before 100% so the decision boundary
      stays smooth across the 4 rule thresholds. Apply to the N_GEN
      synthetic features → labels.

Why this matches the empirical signature of the real synth dataset
  - zero exact feature-vector duplicates in 630k (continuous generator)
  - rule_acc 0.98364 / bal_acc 0.96097 with flips concentrated at the
    boundary scores {3, 6, 7, 8} (smooth NN labeler near rule cuts)
  - flips deterministic in non-rule features (Humidity,
    Previous_Irrigation_mm) — labeler attends to features the rule
    ignores, exactly because the MLP isn't constrained to the rule

CLI
  python3 scripts/reproduce_dgp_v1_sdv_mlp.py [options]

  --n-gen N            number of rows to synthesise (default 50_000;
                       bump to 630_000 for full-scale reproduction)
  --tvae-epochs N      SDV TVAE epochs (default 500)
  --tvae-emb N         TVAE embedding + compress dim (default 256)
  --mlp-epochs N       MLP labeler max epochs (default 250)
  --mlp-stop-acc F     early-stop val_acc on MLP (default 0.985 — keeps
                       the boundary smooth; 1.0 would memorise the rule)
  --mlp-hidden CSV     comma-sep hidden sizes (default 256,192,128)
  --seed S             RNG seed (default 42)
  --skip-tvae          reuse cached TVAE artefact + sample fresh rows
  --no-validate        skip the rule-accuracy / KS / chi-sq report

Outputs
  data/reproduced_v1_train.csv                  (N_GEN rows, comp schema)
  scripts/artifacts/reproduce_v1_tvae.pkl       fitted TVAE synthesiser
  scripts/artifacts/reproduce_v1_validation.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# ----- DGP rule (inlined from scripts/dgp_formula.py to keep self-contained)
ACTIVE_STAGES = ("Flowering", "Vegetative")


def dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"].astype(float).values < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float).values < 300).astype(int)
    hot = (df["Temperature_C"].astype(float).values > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float).values > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(int)
    kc = np.where(np.isin(df["Crop_Growth_Stage"].astype(str).values, ACTIVE_STAGES), 2, 0)
    return 2 * (dry + norain) + (hot + windy + nomulch) + kc


def dgp_predict(df: pd.DataFrame) -> np.ndarray:
    s = dgp_score(df)
    return np.where(s <= 3, "Low", np.where(s <= 6, "Medium", "High"))

# ----- constants ----------------------------------------------------------
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART_DIR = Path("scripts/artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR = Path("data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================ Stage A: TVAE
def fit_tvae(orig_features: pd.DataFrame, epochs: int, emb_dim: int, seed: int):
    """Fit SDV TVAE on the original features (label already dropped)."""
    from sdv.metadata import Metadata
    from sdv.single_table import TVAESynthesizer

    log(f"Stage A: fitting TVAE on {len(orig_features)} original rows × "
        f"{orig_features.shape[1]} features  (epochs={epochs}, emb={emb_dim})")

    metadata = Metadata.detect_from_dataframe(orig_features, table_name="irr")

    syn = TVAESynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=False,
        epochs=epochs,
        batch_size=250,
        embedding_dim=emb_dim,
        compress_dims=(emb_dim, emb_dim),
        decompress_dims=(emb_dim, emb_dim),
        verbose=False,
        cuda=torch.cuda.is_available(),
    )
    t0 = time.time()
    syn.fit(orig_features)
    log(f"  TVAE fit done in {time.time() - t0:.1f}s")
    return syn


def sample_tvae(syn, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows from a fitted TVAE."""
    log(f"Stage A: sampling {n:,} rows from TVAE")
    t0 = time.time()
    # SDV uses the global numpy RNG; seed it for reproducibility.
    np.random.seed(seed)
    df = syn.sample(num_rows=n)
    log(f"  sampled in {time.time() - t0:.1f}s; shape={df.shape}")
    return df


# ============================================================ Stage B: MLP
class TabularMLP(nn.Module):
    """Per-feature embedding (numeric and categorical) + MLP for 3-class.

    Numerics get a Linear(1, num_emb) lift before concat — gives the
    network a learnable basis per feature, which materially helps
    learn axis-aligned thresholds on small training sets (the rule
    we are trying to approximate sits on 4 such thresholds).
    """

    def __init__(self, n_num: int, cat_cards: list[int],
                 hidden=(256, 192, 128),
                 num_emb: int = 8, cat_emb: int = 16,
                 dropout: float = 0.05):
        super().__init__()
        self.num_lifts = nn.ModuleList(
            [nn.Linear(1, num_emb) for _ in range(n_num)])
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(c, cat_emb) for c in cat_cards])
        in_dim = n_num * num_emb + len(cat_cards) * cat_emb
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        num_pieces = [
            lift(x_num[:, i:i + 1]) for i, lift in enumerate(self.num_lifts)]
        cat_pieces = [e(x_cat[:, i]) for i, e in enumerate(self.cat_embs)]
        x = torch.cat(num_pieces + cat_pieces, dim=1)
        return self.net(x)


def encode_tabular(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    cat_vocab: dict[str, dict] | None = None,
    scaler: StandardScaler | None = None,
):
    """Encode (num, cat) → (np.float32, np.int64). Returns also fitted scaler+vocab."""
    if cat_vocab is None:
        cat_vocab = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))}
                     for c in cat_cols}
    cat_arr = np.column_stack([
        df[c].astype(str).map(cat_vocab[c]).fillna(0).astype(np.int64).to_numpy()
        for c in cat_cols
    ]) if cat_cols else np.zeros((len(df), 0), dtype=np.int64)

    num_raw = df[num_cols].astype(np.float32).to_numpy()
    if scaler is None:
        scaler = StandardScaler().fit(num_raw)
    num_arr = scaler.transform(num_raw).astype(np.float32)

    return num_arr, cat_arr, scaler, cat_vocab


def fit_mlp_labeler(
    orig: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    epochs: int,
    stop_val_acc: float,
    hidden: tuple[int, ...],
    seed: int,
):
    """Train MLP on rule-perfect original labels. Stop early at stop_val_acc."""
    log(f"Stage B: training MLP labeler (epochs<={epochs}, "
        f"stop@val_acc>={stop_val_acc:.3f})")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 80/20 split on the 10k for early-stopping signal. Stratify on label.
    rng = np.random.default_rng(seed)
    idx = np.arange(len(orig))
    perm = rng.permutation(idx)
    split = int(0.8 * len(orig))
    tr_idx, va_idx = perm[:split], perm[split:]

    num_tr, cat_tr, scaler, vocab = encode_tabular(
        orig.iloc[tr_idx], num_cols, cat_cols)
    num_va, cat_va, _, _ = encode_tabular(
        orig.iloc[va_idx], num_cols, cat_cols, cat_vocab=vocab, scaler=scaler)

    y_tr = orig.iloc[tr_idx][TARGET].map(CLS2IDX).to_numpy().astype(np.int64)
    y_va = orig.iloc[va_idx][TARGET].map(CLS2IDX).to_numpy().astype(np.int64)

    cards = [len(vocab[c]) for c in cat_cols]
    model = TabularMLP(
        n_num=len(num_cols), cat_cards=cards, hidden=tuple(hidden),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def to_t(x_n, x_c, y=None):
        a = torch.from_numpy(x_n).to(DEVICE)
        b = torch.from_numpy(x_c).to(DEVICE)
        if y is None:
            return a, b
        return a, b, torch.from_numpy(y).to(DEVICE)

    Ntr = len(tr_idx)
    bs = 256
    best_state, best_val = None, -1.0
    stop_epoch = epochs
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(Ntr)
        for s in range(0, Ntr, bs):
            ix = order[s:s + bs]
            xn, xc, yy = to_t(num_tr[ix], cat_tr[ix], y_tr[ix])
            logits = model(xn, xc)
            loss = F.cross_entropy(logits, yy)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            xn, xc = to_t(num_va, cat_va)
            pred = model(xn, xc).argmax(dim=1).cpu().numpy()
        val_acc = float((pred == y_va).mean())

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == 1:
            log(f"  epoch {ep:3d}  val_acc={val_acc:.4f}  (best={best_val:.4f})")

        if val_acc >= stop_val_acc:
            log(f"  early-stop at epoch {ep}, val_acc={val_acc:.4f} ≥ "
                f"{stop_val_acc:.3f}")
            stop_epoch = ep
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, vocab, dict(best_val=best_val, stop_epoch=stop_epoch)


def label_with_mlp(
    model: TabularMLP,
    feats: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    scaler: StandardScaler,
    vocab: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply trained MLP to feats. Returns (labels_str, probs)."""
    log(f"Stage B: labelling {len(feats):,} synthesised rows")
    num_a, cat_a, _, _ = encode_tabular(
        feats, num_cols, cat_cols, cat_vocab=vocab, scaler=scaler)
    model.eval()
    with torch.no_grad():
        bs = 8192
        probs = np.zeros((len(feats), 3), dtype=np.float32)
        for s in range(0, len(feats), bs):
            xn = torch.from_numpy(num_a[s:s + bs]).to(DEVICE)
            xc = torch.from_numpy(cat_a[s:s + bs]).to(DEVICE)
            logits = model(xn, xc)
            probs[s:s + bs] = F.softmax(logits, dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    labels = np.array([IDX2CLS[i] for i in pred_idx])
    return labels, probs


# =========================================================== Validation
def validate_against_real(repr_df: pd.DataFrame, real_train: pd.DataFrame,
                          num_cols: list[str], cat_cols: list[str]) -> dict:
    """Side-by-side comparison of reproduced dataset vs real synth train."""
    log("validating reproduced dataset vs real synth train")

    out: dict = {}

    # 1) class prior
    prior_real = real_train[TARGET].value_counts(normalize=True).reindex(CLASSES).round(5).to_dict()
    prior_repr = repr_df[TARGET].value_counts(normalize=True).reindex(CLASSES).round(5).to_dict()
    out["class_prior_real"] = prior_real
    out["class_prior_repr"] = prior_repr

    # 2) rule accuracy + bal_acc on the reproduced set (real synth: 0.98364 / 0.96097)
    rule_pred = dgp_predict(repr_df)
    y_repr = repr_df[TARGET].to_numpy()
    rule_acc_repr = float((rule_pred == y_repr).mean())
    rule_balacc_repr = float(balanced_accuracy_score(
        [CLS2IDX[c] for c in y_repr], [CLS2IDX[c] for c in rule_pred]))
    out["rule_acc_real"] = 0.98364
    out["rule_acc_repr"] = round(rule_acc_repr, 5)
    out["rule_balacc_real"] = 0.96097
    out["rule_balacc_repr"] = round(rule_balacc_repr, 5)

    # 3) per-score error rate (where the flips concentrate)
    score_repr = dgp_score(repr_df)
    score_real_table = {  # from CLAUDE.md 2026-04-21 score-routing analysis
        0: 0.0000, 1: 0.00004, 2: 0.00299, 3: 0.0480, 4: 0.0129,
        5: 0.00350, 6: 0.0403, 7: 0.0905, 8: 0.1231, 9: 0.00062,
    }
    err_by_score_repr = {}
    for s in range(10):
        m = score_repr == s
        if m.sum() == 0:
            err_by_score_repr[s] = None
        else:
            err_by_score_repr[s] = round(float((rule_pred[m] != y_repr[m]).mean()), 5)
    out["err_by_score_real"] = score_real_table
    out["err_by_score_repr"] = err_by_score_repr

    # 4) numeric KS distances
    ks = {}
    for c in num_cols:
        stat, _ = stats.ks_2samp(real_train[c].to_numpy(), repr_df[c].to_numpy())
        ks[c] = round(float(stat), 4)
    out["ks_per_numeric"] = ks
    out["ks_mean"] = round(float(np.mean(list(ks.values()))), 4)

    # 5) categorical chi-square (categories already aligned by schema)
    chi = {}
    for c in cat_cols:
        a = real_train[c].value_counts().reindex(sorted(real_train[c].unique())).fillna(0)
        b = repr_df[c].value_counts().reindex(a.index).fillna(0)
        # normalise to expected counts under real's frequency, scale by repr size
        exp = a / a.sum() * b.sum()
        chi_stat, p = stats.chisquare(b.to_numpy(), exp.to_numpy())
        chi[c] = dict(chi2=round(float(chi_stat), 2), p=round(float(p), 4))
    out["chi2_per_cat"] = chi

    return out


# =========================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=50_000)
    ap.add_argument("--tvae-epochs", type=int, default=500)
    ap.add_argument("--tvae-emb", type=int, default=256)
    ap.add_argument("--mlp-epochs", type=int, default=250)
    ap.add_argument("--mlp-stop-acc", type=float, default=0.985)
    ap.add_argument("--mlp-hidden", type=str, default="256,192,128",
                    help="comma-sep hidden layer sizes for the MLP labeler")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-tvae", action="store_true",
                    help="reuse cached TVAE; only re-sample features")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--out-tag", type=str, default="",
                    help="suffix for outputs (csv + validation json), so "
                         "successive configs don't overwrite each other")
    args = ap.parse_args()
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(","))
    tag = f"_{args.out_tag}" if args.out_tag else ""

    log(f"=== Option 1: SDV TVAE + MLP labeler  (n_gen={args.n_gen})")
    log(f"device={DEVICE}  seed={args.seed}")

    orig = pd.read_csv("data/archive.zip")
    log(f"loaded original: {orig.shape}")

    num_cols = [c for c in orig.select_dtypes(include=np.number).columns if c != TARGET]
    cat_cols = [c for c in orig.columns if c not in num_cols + [TARGET]]
    log(f"  num_cols ({len(num_cols)}): {num_cols}")
    log(f"  cat_cols ({len(cat_cols)}): {cat_cols}")

    # ----- Stage A: TVAE on features only -----------------------------------
    tvae_path = ART_DIR / "reproduce_v1_tvae.pkl"
    feats_only = orig.drop(columns=[TARGET])

    if args.skip_tvae and tvae_path.exists():
        from sdv.single_table import TVAESynthesizer
        log(f"loading cached TVAE from {tvae_path}")
        syn = TVAESynthesizer.load(tvae_path)
    else:
        syn = fit_tvae(feats_only, epochs=args.tvae_epochs,
                       emb_dim=args.tvae_emb, seed=args.seed)
        syn.save(tvae_path)
        log(f"  cached TVAE → {tvae_path}")

    feats_gen = sample_tvae(syn, n=args.n_gen, seed=args.seed)

    # ----- Stage B: MLP labeler ---------------------------------------------
    mlp, scaler, vocab, mlp_meta = fit_mlp_labeler(
        orig, num_cols, cat_cols,
        epochs=args.mlp_epochs, stop_val_acc=args.mlp_stop_acc,
        hidden=mlp_hidden, seed=args.seed)

    labels, probs = label_with_mlp(mlp, feats_gen, num_cols, cat_cols, scaler, vocab)
    repr_df = feats_gen.copy()
    repr_df[TARGET] = labels

    # ----- write reproduced dataset -----------------------------------------
    out_csv = DATA_DIR / f"reproduced_v1{tag}_train.csv"
    repr_df.to_csv(out_csv, index=False)
    log(f"wrote {out_csv}  shape={repr_df.shape}")

    # ----- validation -------------------------------------------------------
    val_report: dict = {
        "args": vars(args),
        "mlp_meta": mlp_meta,
        "n_gen": int(args.n_gen),
    }
    if not args.no_validate:
        real_train = pd.read_csv("data/train.csv")
        val_report["validation"] = validate_against_real(
            repr_df, real_train, num_cols, cat_cols)

    val_path = ART_DIR / f"reproduce_v1{tag}_validation.json"
    with open(val_path, "w") as f:
        json.dump(val_report, f, indent=2, default=str)
    log(f"wrote {val_path}")

    if "validation" in val_report:
        v = val_report["validation"]
        print()
        print("=== Reproduction quality summary ===")
        print(f"  rule_acc     real={v['rule_acc_real']:.5f}  "
              f"repr={v['rule_acc_repr']:.5f}  "
              f"Δ={v['rule_acc_repr'] - v['rule_acc_real']:+.5f}")
        print(f"  rule_bal_acc real={v['rule_balacc_real']:.5f}  "
              f"repr={v['rule_balacc_repr']:.5f}  "
              f"Δ={v['rule_balacc_repr'] - v['rule_balacc_real']:+.5f}")
        print(f"  KS mean (numeric marginals)    : {v['ks_mean']:.4f}")
        print(f"  class prior real: {v['class_prior_real']}")
        print(f"  class prior repr: {v['class_prior_repr']}")
        print()
        print("  per-score error rate (real vs repr):")
        for s in range(10):
            r = v["err_by_score_real"][s]
            p = v["err_by_score_repr"][s]
            p_str = f"{p:.4f}" if p is not None else "  n/a"
            print(f"    score {s}:  real={r:.4f}   repr={p_str}")


if __name__ == "__main__":
    main()
