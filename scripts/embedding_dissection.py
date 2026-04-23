"""Embedding-space dissection for Exp 6 baseline.

Extracts context-encoder embeddings for multiple clips per recording on the
validation set, then quantifies how much of the embedding space is driven by
subject identity vs movie-time/stimulus.

Outputs (saved to --out_dir):
  - embeddings.npz       raw data: emb [N,D], rec_id [N], pos [N], lum [N],
                         con [N], nar [N], age [N], sex [N]
  - report.json          numeric results for the table at the bottom
  - report.md            human-readable summary
  - per_pc.png           bar plot of per-PC variance fractions vs. subject/stim

Usage on Delta:
  PYTHONPATH=. uv run --group eeg python scripts/embedding_dissection.py \
      --checkpoint=/path/to/best.pth.tar \
      --corrca_filters=corrca_filters.npz \
      --out_dir=embedding_dissection_exp6 \
      --clips_per_rec=8
"""

import json
from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from eb_jepa.architectures import EEGEncoderTokens
from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.training_utils import load_config, setup_device
from experiments.eeg_jepa.main import resolve_preprocessed_dir


FEATURE_NAMES = ["contrast_rms", "luminance_mean", "position_in_movie", "narrative_event_score"]


def load_encoder(ckpt_path, cfg, chs_info, n_chans, n_times, device):
    encoder = EEGEncoderTokens(
        n_chans=n_chans,
        n_times=n_times,
        embed_dim=cfg.model.encoder_embed_dim,
        depth=cfg.model.encoder_depth,
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        n_windows=cfg.data.n_windows,
        patch_size=cfg.model.get("patch_size", 200),
        patch_overlap=cfg.model.get("patch_overlap", 20),
        freqs=cfg.model.get("freqs", 4),
        chs_info=chs_info,
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prefix = "context_encoder."
    sd = {k[len(prefix):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    print(f"Loaded encoder: {len(sd)} weights. Missing={len(missing)}, unexpected={len(unexpected)}")
    return encoder.to(device).eval()


@torch.no_grad()
def extract_clip_embeddings(dataset, encoder, device, clips_per_rec=8):
    """Per-clip mean-pooled embeddings across val recordings.

    Returns dict with arrays keyed by clip index.
    """
    embs, rec_ids, feats = [], [], []
    ages, sexes = [], []
    n_rec = len(dataset)

    for rec_idx in range(n_rec):
        crop_inds = dataset._crop_inds[rec_idx]
        required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
        n_clips_total = len(crop_inds) - required + 1
        if n_clips_total <= 0:
            continue

        n_sample = min(clips_per_rec, n_clips_total)
        starts = np.linspace(0, n_clips_total - 1, n_sample, dtype=int)

        meta = dataset._recording_metadata[rec_idx]
        try:
            age = float(meta.get("age", np.nan))
        except (TypeError, ValueError):
            age = np.nan
        sex_raw = str(meta.get("sex", meta.get("gender", ""))).strip().lower()
        sex = 1.0 if sex_raw in ("m", "male") else (0.0 if sex_raw in ("f", "female") else np.nan)

        for start in starts:
            window_idx = list(range(start, start + required, dataset.temporal_stride))
            eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[window_idx])
            eeg = torch.from_numpy(eeg_np)

            if dataset._norm_mode == "per_recording":
                m = eeg.mean(dim=(0, 2), keepdim=True)
                s = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
                eeg = (eeg - m) / s
            else:
                eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
            if dataset._add_envelope:
                eeg = dataset._append_lowfreq_envelope(eeg)
            if dataset._corrca_W is not None:
                eeg = torch.einsum("wct,ck->wkt", eeg, dataset._corrca_W)

            eeg = eeg.unsqueeze(0).to(device)
            tokens = encoder.encode_tokens(eeg, mask=None)  # [1, T, D]
            emb = tokens.mean(dim=1).squeeze(0).cpu().numpy()

            # Per-clip features: mean over the clip's windows.
            # feature_recordings[rec_idx] has shape [n_win, n_features] where
            # feature_names = [contrast_rms, luminance_mean, position_in_movie,
            # narrative_event_score].
            feat = dataset.feature_recordings[rec_idx][window_idx].mean(dim=0).numpy()

            embs.append(emb)
            rec_ids.append(rec_idx)
            feats.append(feat)
            ages.append(age)
            sexes.append(sex)

        if (rec_idx + 1) % 20 == 0:
            print(f"  processed {rec_idx + 1}/{n_rec} recordings, {len(embs)} clips")

    return {
        "emb": np.stack(embs).astype(np.float32),
        "rec_id": np.array(rec_ids, dtype=np.int64),
        "feat": np.stack(feats).astype(np.float32),
        "age": np.array(ages, dtype=np.float32),
        "sex": np.array(sexes, dtype=np.float32),
    }


def variance_decomposition(emb, rec_id):
    """Between-recording vs within-recording variance per dim.

    Returns: (var_between [D], var_within [D], var_total [D])
    """
    D = emb.shape[1]
    unique_recs = np.unique(rec_id)
    rec_means = np.stack([emb[rec_id == r].mean(axis=0) for r in unique_recs])  # [R, D]
    global_mean = emb.mean(axis=0)  # [D]

    # weighted between-recording variance (weight by n clips per rec)
    weights = np.array([(rec_id == r).sum() for r in unique_recs], dtype=np.float32)
    w = weights / weights.sum()
    var_between = (w[:, None] * (rec_means - global_mean) ** 2).sum(axis=0)

    var_total = emb.var(axis=0)
    var_within = np.maximum(var_total - var_between, 0.0)
    return var_between, var_within, var_total


def fit_per_pc_probes(pcs, rec_id, feat, age, sex):
    """For each PC, fit univariate linear probes.

    Returns a dict {metric: [D] array}.
    """
    D = pcs.shape[1]
    results = {
        "subj_acc_top1": np.zeros(D),    # train=test subject-id linear (sanity upper bound)
        "age_r2": np.full(D, np.nan),
        "sex_auc": np.full(D, np.nan),
        "pos_r2": np.full(D, np.nan),
        "lum_r2": np.full(D, np.nan),
        "con_r2": np.full(D, np.nan),
        "nar_r2": np.full(D, np.nan),
    }

    # Subject ID as multiclass label — use rec_id directly
    # (each recording is one subject here; we probe "identifiability")
    for d in range(D):
        x = pcs[:, d:d+1]

        # ---- age (regression on recording-level mean) ----
        mask = ~np.isnan(age)
        if mask.sum() > 10 and np.std(age[mask]) > 0:
            r = Ridge(alpha=1.0).fit(x[mask], age[mask])
            results["age_r2"][d] = r2_score(age[mask], r.predict(x[mask]))

        # ---- sex ----
        mask = ~np.isnan(sex)
        if mask.sum() > 10 and len(np.unique(sex[mask])) == 2:
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(x[mask], sex[mask])
            probs = clf.predict_proba(x[mask])[:, 1]
            results["sex_auc"][d] = roc_auc_score(sex[mask], probs)

        # ---- stimulus features ----
        for key, col in [("pos_r2", 2), ("lum_r2", 1), ("con_r2", 0), ("nar_r2", 3)]:
            y = feat[:, col]
            r = Ridge(alpha=1.0).fit(x, y)
            results[key][d] = r2_score(y, r.predict(x))

    # Subject identifiability via single-PC classifier is too weak (hundreds of classes).
    # Instead use a simple metric: how well a linear Ridge on this PC predicts rec_id's mean embedding on other dims (not meaningful here).
    # Leave subj_acc as 0 for now; we use between-recording variance as the main subject proxy.

    return results


def train_split_probes(pcs_train, labels_train, pcs_val, labels_val, task="reg"):
    """Cumulative probe: train on top-k PCs, eval on val.

    Returns dict {k: metric}.
    """
    D = pcs_train.shape[1]
    out = {}
    for k in [1, 2, 5, 10, 20, min(40, D), D]:
        if k > D:
            continue
        Xt = pcs_train[:, :k]
        Xv = pcs_val[:, :k]
        if task == "reg":
            m = Ridge(alpha=1.0).fit(Xt, labels_train)
            out[k] = r2_score(labels_val, m.predict(Xv))
        else:
            if len(np.unique(labels_train)) != 2:
                return out
            m = LogisticRegression(max_iter=1000, C=1.0).fit(Xt, labels_train)
            probs = m.predict_proba(Xv)[:, 1]
            out[k] = roc_auc_score(labels_val, probs)
    return out


def run(
    checkpoint: str,
    config: str = "experiments/eeg_jepa/cfgs/default.yaml",
    out_dir: str = "embedding_dissection",
    clips_per_rec: int = 8,
    norm_mode: str = "per_recording",
    corrca_filters: str | None = None,
    preprocessed_dir: str | None = None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    overrides = {
        "model.encoder_depth": 2,
        "data.norm_mode": norm_mode,
    }
    if corrca_filters is not None:
        overrides["data.corrca_filters"] = corrca_filters
    cfg = load_config(config, overrides)
    device = setup_device(cfg.meta.device)

    print("Loading val dataset...")
    resolved_prep = resolve_preprocessed_dir(preprocessed_dir or cfg.data.get("preprocessed_dir", None))
    val_set = JEPAMovieDataset(
        split="val",
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        cfg=cfg.data,
        preprocessed=cfg.data.get("preprocessed", True) or resolved_prep is not None,
        preprocessed_dir=resolved_prep,
    )
    print(f"  {len(val_set)} val recordings, {val_set.n_chans} channels")

    print(f"Loading encoder from {checkpoint}...")
    encoder = load_encoder(
        checkpoint, cfg,
        chs_info=val_set.get_chs_info(),
        n_chans=val_set.n_chans,
        n_times=val_set.n_times,
        device=device,
    )

    print(f"Extracting embeddings ({clips_per_rec} clips/rec)...")
    data = extract_clip_embeddings(val_set, encoder, device, clips_per_rec=clips_per_rec)
    np.savez(out / "embeddings.npz", **data)
    print(f"  saved {data['emb'].shape[0]} clips × {data['emb'].shape[1]} dims")

    emb = data["emb"]
    rec_id = data["rec_id"]
    feat = data["feat"]
    age = data["age"]
    sex = data["sex"]

    # --- Variance decomposition per native dim ---
    var_b, var_w, var_t = variance_decomposition(emb, rec_id)
    frac_between = var_b / (var_t + 1e-8)
    print(f"Native-dim variance: between-rec={var_b.sum():.3f}, "
          f"within-rec={var_w.sum():.3f}, total={var_t.sum():.3f}")
    print(f"Between-rec fraction (pooled): {var_b.sum() / (var_t.sum() + 1e-8):.4f}")

    # --- PCA ---
    pca = PCA(n_components=min(emb.shape[1], 64))
    pcs = pca.fit_transform(emb)
    explained = pca.explained_variance_ratio_
    print(f"PCA: top-5 explained = {explained[:5].round(3).tolist()}, "
          f"top-10 cum = {explained[:10].sum():.3f}")

    # --- Variance decomposition per PC ---
    pc_var_b, pc_var_w, pc_var_t = variance_decomposition(pcs, rec_id)
    pc_frac_between = pc_var_b / (pc_var_t + 1e-8)

    # --- Per-PC single-dim probes ---
    probes = fit_per_pc_probes(pcs, rec_id, feat, age, sex)

    # --- Cumulative probes (train/val split by recording) ---
    unique_recs = np.unique(rec_id)
    rng = np.random.default_rng(0)
    perm = rng.permutation(unique_recs)
    n_train = int(0.7 * len(perm))
    train_recs = set(perm[:n_train].tolist())
    train_mask = np.array([r in train_recs for r in rec_id])

    # scale embeddings (fit on train)
    scaler = StandardScaler().fit(pcs[train_mask])
    pcs_s = scaler.transform(pcs)

    cumulative = {}
    for key, y, task in [
        ("position_in_movie", feat[:, 2], "reg"),
        ("luminance_mean", feat[:, 1], "reg"),
        ("contrast_rms", feat[:, 0], "reg"),
        ("narrative_event", feat[:, 3], "reg"),
        ("age", age, "reg"),
        ("sex", sex, "cls"),
    ]:
        val_mask = (~train_mask) & ~np.isnan(y)
        tr_mask = train_mask & ~np.isnan(y)
        if tr_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        cumulative[key] = train_split_probes(
            pcs_s[tr_mask], y[tr_mask], pcs_s[val_mask], y[val_mask], task=task
        )

    # --- Dump report ---
    report = {
        "n_clips": int(len(emb)),
        "n_recordings": int(len(unique_recs)),
        "embed_dim": int(emb.shape[1]),
        "pooled_frac_between_recording": float(var_b.sum() / (var_t.sum() + 1e-8)),
        "pca_top_explained_variance": explained[:20].tolist(),
        "pca_cum_explained_top10": float(explained[:10].sum()),
        "per_pc_frac_between_recording": pc_frac_between[:20].tolist(),
        "per_pc_probes": {k: v[:20].tolist() for k, v in probes.items()},
        "cumulative_topk_probes": cumulative,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    k = 20
    axes[0].bar(range(k), explained[:k])
    axes[0].set_title("PC explained variance (top 20)")
    axes[0].set_xlabel("PC index")
    axes[0].set_ylabel("Explained var ratio")

    axes[1].bar(range(k), pc_frac_between[:k], color="crimson", label="between-rec")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.5, color="gray", ls="--")
    axes[1].set_title("Per-PC between-rec variance fraction")
    axes[1].set_xlabel("PC index")

    probe_pos = probes["pos_r2"][:k]
    probe_lum = probes["lum_r2"][:k]
    probe_sex = probes["sex_auc"][:k] - 0.5  # offset so chance=0
    probe_age = probes["age_r2"][:k]

    axes[2].plot(range(k), probe_pos, "o-", label="position R²")
    axes[2].plot(range(k), probe_lum, "s-", label="luminance R²")
    axes[2].plot(range(k), probe_age, "d-", label="age R²")
    axes[2].plot(range(k), probe_sex, "^-", label="sex AUC−0.5")
    axes[2].axhline(0, color="gray", ls="--")
    axes[2].set_title("Per-PC probe signal")
    axes[2].set_xlabel("PC index")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out / "per_pc.png", dpi=150)
    print(f"Saved per_pc.png")

    # --- Human-readable summary ---
    md = []
    md.append("# Embedding-space dissection report\n")
    md.append(f"Checkpoint: `{checkpoint}`\n")
    md.append(f"- {len(emb)} val clips across {len(unique_recs)} recordings\n")
    md.append(f"- Embedding dim: {emb.shape[1]}\n\n")
    md.append("## Variance decomposition (pooled)\n")
    md.append(f"- Between-recording variance / total: **{var_b.sum() / (var_t.sum()+1e-8):.3f}**\n")
    md.append(f"  (higher = more subject structure; 0 = pure stimulus)\n\n")
    md.append("## PCA\n")
    md.append(f"- Top-10 PCs explain {explained[:10].sum():.3f} of variance\n")
    md.append(f"- PC1-5 explained: {[round(x, 3) for x in explained[:5]]}\n\n")
    md.append("## Per-PC between-rec fraction (top 20)\n")
    md.append("|PC|explained|between-rec frac|pos R²|lum R²|age R²|sex AUC|\n")
    md.append("|--|--|--|--|--|--|--|\n")
    for i in range(min(20, len(explained))):
        md.append(
            f"|{i}|{explained[i]:.3f}|{pc_frac_between[i]:.2f}|"
            f"{probes['pos_r2'][i]:.3f}|{probes['lum_r2'][i]:.3f}|"
            f"{probes['age_r2'][i]:.3f}|{probes['sex_auc'][i]:.3f}|\n"
        )
    md.append("\n## Cumulative top-k probes (70/30 split)\n")
    for metric, kv in cumulative.items():
        md.append(f"- **{metric}**: {kv}\n")
    (out / "report.md").write_text("".join(md))
    print(f"Saved report.md")


if __name__ == "__main__":
    fire.Fire(run)
