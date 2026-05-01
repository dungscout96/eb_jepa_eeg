"""Linear probe of trivial within-clip EEG features → movie features.

Tests whether position_in_movie (and luminance, contrast, narrative) can
be predicted from simple temporal features that don't require a learned
encoder — channel mean, std, log bandpower in standard bands. If position
is well-decoded by trivial features but luminance/contrast aren't, the
encoder's "position signal" is likely a within-recording temporal-trend
artifact, not stimulus content.

Setup mirrors probe_eval.py exactly (per-rec norm, optional CorrCA,
matching n_windows / window_size_seconds), so the trivial features see the
same EEG the encoder sees.

Usage
-----
PYTHONPATH=. uv run --group eeg python scripts/trivial_position_baseline.py \\
    --norm_mode=per_recording \\
    --corrca_filters=/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz \\
    --n_windows=4 --window_size_seconds=2 \\
    --n_passes=20 --seed=2025
"""

from pathlib import Path

import fire
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.signal import butter, filtfilt, welch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from eb_jepa.datasets.hbn import (
    JEPAMovieDataset, MOVIE_METADATA, _read_raw_windows,
)
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_seed
from experiments.eeg_jepa.main import resolve_preprocessed_dir

logger = get_logger(__name__)

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


def _bandpower(eeg: np.ndarray, sfreq: float) -> np.ndarray:
    """Per-channel log power in 5 bands.

    eeg: [n_windows, C, T] → returns [C * 5] feature vector.
    """
    flat = eeg.reshape(-1, eeg.shape[-1])  # [n_windows*C, T]
    nperseg = min(256, eeg.shape[-1])
    f, psd = welch(flat, fs=sfreq, nperseg=nperseg, axis=-1)
    out = []
    for lo, hi in BANDS.values():
        mask = (f >= lo) & (f < hi)
        bp = np.log(psd[:, mask].mean(axis=-1) + 1e-12)  # [n_windows*C]
        bp = bp.reshape(eeg.shape[0], eeg.shape[1]).mean(axis=0)  # avg across windows → [C]
        out.append(bp)
    return np.concatenate(out)  # [C*5]


def _trivial_features(eeg: torch.Tensor, sfreq: float) -> np.ndarray:
    """Compute trivial within-clip features. eeg: [n_windows, C, T]."""
    eeg_np = eeg.numpy()
    means = eeg_np.mean(axis=(0, 2))           # [C]
    stds = eeg_np.std(axis=(0, 2))             # [C]
    bp = _bandpower(eeg_np, sfreq)             # [C*5]
    return np.concatenate([means, stds, bp])   # [C*7]


def _jepa_features(eeg: torch.Tensor, jepa, device, keep_channels: bool) -> np.ndarray:
    """Encode a clip with JEPA → mean across windows → [D] vector."""
    # eeg [nw,C,T]; jepa.encode expects [B, nw, C, T].
    with torch.no_grad():
        x = eeg.unsqueeze(0).to(device)
        state = jepa.encode(x, keep_channels=keep_channels)  # [1, D, nw, 1, 1]
        emb = state.squeeze(0).mean(dim=(1, 2, 3)).cpu().numpy()  # [D]
    return emb


def _extract(dataset, n_passes, sfreq, seed, group_by_rec: bool = False,
             jepa=None, device=None, keep_channels: bool = False):
    """Iterate dataset n_passes times; collect features + labels.

    Features: trivial mean+std+log-band stats (default) or JEPA per-clip
    embedding (when ``jepa`` is provided). The JEPA path matches the draft's
    ridge protocol but with encoder-derived features in place of trivial
    stats.

    If group_by_rec, output shape is (n_rec, n_passes, ...) so the bootstrap
    can index recordings on axis 0 (matches probe_eval.py npz schema).
    """
    use_jepa = jepa is not None

    def _feat(eeg):
        if use_jepa:
            return _jepa_features(eeg, jepa, device, keep_channels)
        return _trivial_features(eeg, sfreq)

    rng = torch.Generator().manual_seed(seed)
    n_rec = len(dataset)
    if group_by_rec:
        # Per-rec sample n_passes clips (random)
        feats_arr = []
        labels_arr = []
        for rec_idx in range(n_rec):
            f_passes = []
            l_passes = []
            for _ in range(n_passes):
                eeg, feats, _ = dataset[rec_idx]
                f_passes.append(_feat(eeg))
                l_passes.append(feats.mean(dim=0).numpy())
            feats_arr.append(np.stack(f_passes))   # [n_passes, D]
            labels_arr.append(np.stack(l_passes))  # [n_passes, n_features]
        X = np.stack(feats_arr)   # [n_rec, n_passes, D]
        Y = np.stack(labels_arr)  # [n_rec, n_passes, n_features]
        return X, Y
    feats_list = []
    labels_list = []
    for p in range(n_passes):
        for rec_idx in torch.randperm(n_rec, generator=rng).tolist():
            eeg, feats, _ = dataset[rec_idx]  # eeg [nw,C,T], feats [nw, n_features]
            feats_list.append(_feat(eeg))
            labels_list.append(feats.mean(dim=0).numpy())  # [n_features]
        if (p + 1) % 5 == 0 or p == n_passes - 1:
            logger.info("  pass %d/%d", p + 1, n_passes)
    X = np.stack(feats_list)
    Y = np.stack(labels_list)
    return X, Y


def _extract_time_aligned(dataset, K, sfreq):
    """K evenly-spaced (linspace) clips per recording — approximately
    time-aligned across recordings since all recordings cover the same movie.
    Bypasses dataset.__getitem__'s random clip sampling.
    """
    feats_list = []
    labels_list = []
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        n_clips = n_total - required + 1
        if n_clips <= 0:
            continue
        starts = np.linspace(0, n_clips - 1, min(K, n_clips), dtype=int)
        for start in starts:
            indices = list(range(start, start + required, dataset.temporal_stride))
            eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
            eeg = torch.from_numpy(eeg_np)
            if dataset._norm_mode == "per_recording":
                rec_mean = eeg.mean(dim=(0, 2), keepdim=True)
                rec_std = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
                eeg = (eeg - rec_mean) / rec_std
            else:
                eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
            if dataset._add_envelope:
                eeg = dataset._append_lowfreq_envelope(eeg)
            if dataset._corrca_W is not None:
                eeg = torch.einsum("wct,ck->wkt", eeg, dataset._corrca_W)
            feats_list.append(_trivial_features(eeg, sfreq))
            labels_list.append(
                dataset.feature_recordings[rec_idx][indices].mean(dim=0).numpy()
            )
    return np.stack(feats_list), np.stack(labels_list)


def _shuffle_label_globally(datasets, target_label, feature_names, seed,
                            movie="ThePresent"):
    """Permute one label column across movie frames, consistently across
    all recordings. At each movie frame f, assign the original label value
    from a random other frame: new_label[f] = original_label[perm[f]].

    Recover the original frame for each window from position_in_movie
    (= frame_idx / (n_frames-1)).
    """
    import pandas as pd
    target_idx = feature_names.index(target_label)
    pos_idx = feature_names.index("position_in_movie")
    parquet_path = MOVIE_METADATA[movie]["feature_parquet"]
    df = pd.read_parquet(parquet_path)
    n_frames = len(df)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_frames)
    permuted_label_at_frame = df[target_label].values[perm]

    for ds in datasets:
        for rec_idx in range(len(ds._fif_paths)):
            feats = ds.feature_recordings[rec_idx]
            positions = feats[:, pos_idx].numpy()
            frame_indices = np.clip(
                np.round(positions * (n_frames - 1)).astype(int), 0, n_frames - 1
            )
            new_vals = permuted_label_at_frame[frame_indices].astype(np.float32)
            feats[:, target_idx] = torch.from_numpy(new_vals)


def run(
    n_windows: int = 4,
    window_size_seconds: int = 2,
    norm_mode: str = "per_recording",
    corrca_filters: str = "",
    n_passes: int = 20,
    seed: int = 2025,
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    # Test 1: globally permute a label column across movie frames
    shuffle_label: str = "",
    # Test 2: K evenly-spaced clips per recording (deterministic, not random)
    time_aligned_K: int = 0,
    # Multi-seed bootstrap inputs: dump per-recording preds (n_rec, n_passes, F)
    save_predictions_dir: str = "",
    output_json: str = "",
    baseline: str = "trivial_ridge",
    # JEPA mode — feed encoder embeddings into the ridge probe instead of
    # handcrafted trivial stats. Requires a checkpoint.
    checkpoint: str = "",
    keep_channels: bool = False,
):
    setup_seed(seed)
    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.norm_mode": norm_mode,
    }
    if corrca_filters:
        overrides["data.corrca_filters"] = corrca_filters
    cfg = load_config(fname, overrides)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feature_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))

    logger.info("Loading datasets (n_passes=%d)...", n_passes)
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feature_names,
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )
    eval_sets = {}
    for split in ("val", "test"):
        eval_sets[split] = JEPAMovieDataset(
            split=split,
            n_windows=n_windows,
            window_size_seconds=window_size_seconds,
            feature_names=feature_names,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
            cfg=cfg.data,
            preprocessed=preprocessed,
            preprocessed_dir=preprocessed_dir,
        )

    sfreq = train_set.n_times / window_size_seconds
    logger.info("sfreq=%.1f Hz, n_chans=%d", sfreq, train_set.n_chans)

    # ----------------------------------------------------------------
    # Optional JEPA encoder (for the JEPA-into-ridge column comparison).
    # When --checkpoint is set, _extract uses encoder embeddings (mean over
    # windows) instead of trivial mean+std+log-band stats.
    # ----------------------------------------------------------------
    jepa_model = None
    jepa_device = None
    if checkpoint:
        import copy
        from eb_jepa.architectures import (
            EEGEncoderTokens, MaskedPredictor,
        )
        from eb_jepa.jepa import MaskedJEPA
        from eb_jepa.training_utils import setup_device
        from eb_jepa.masking import MultiBlockMaskCollator
        jepa_device = setup_device("auto")
        sd_dict = torch.load(checkpoint, map_location=jepa_device, weights_only=False)
        sd = sd_dict.get("model_state_dict", sd_dict)
        # Discover predictor bottleneck dim from the saved state dict
        try:
            _pred_dim = int(sd["predictor.input_proj.weight"].shape[0])
        except KeyError:
            try:
                _pred_dim = int(sd["predictor.output_proj.weight"].shape[1])
            except KeyError:
                _pred_dim = None
        encoder = EEGEncoderTokens(
            n_chans=train_set.n_chans,
            n_times=train_set.n_times,
            embed_dim=cfg.model.encoder_embed_dim,
            depth=cfg.model.encoder_depth,
            heads=cfg.model.encoder_heads,
            head_dim=cfg.model.encoder_head_dim,
            n_windows=n_windows,
            patch_size=cfg.model.get("patch_size", 50),
            patch_overlap=cfg.model.get("patch_overlap", 20),
            freqs=cfg.model.get("freqs", 4),
            chs_info=train_set.get_chs_info(),
            mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
        ).to(jepa_device)
        target_encoder = copy.deepcopy(encoder)
        predictor = MaskedPredictor(
            encoder.embed_dim,
            depth=cfg.model.predictor_depth,
            heads=cfg.model.predictor_heads,
            head_dim=cfg.model.predictor_head_dim,
            embed_dim=_pred_dim if _pred_dim else cfg.model.encoder_embed_dim,
        ).to(jepa_device)
        masking_cfg = cfg.masking
        mask_collator = MultiBlockMaskCollator(
            n_chans=train_set.n_chans,
            n_patches_per_window=encoder.n_patches_per_window,
            mask_config=masking_cfg,
        )
        from eb_jepa.losses import VCLoss
        jepa_model = MaskedJEPA(
            encoder, target_encoder, predictor, mask_collator,
            VCLoss(std_coeff=0.0, cov_coeff=0.0),
            n_pred_masks_short=masking_cfg.get("n_pred_masks_short", 2),
            n_pred_masks_long=masking_cfg.get("n_pred_masks_long", 2),
        ).to(jepa_device)
        missing, unexpected = jepa_model.load_state_dict(sd, strict=False)
        jepa_model.eval()
        logger.info(
            "Loaded JEPA from %s (missing=%d, unexpected=%d, keep_channels=%s)",
            checkpoint, len(missing), len(unexpected), keep_channels,
        )

    if shuffle_label:
        logger.info("Shuffling label '%s' globally across movie frames (seed=%d)",
                    shuffle_label, seed)
        _shuffle_label_globally(
            [train_set, eval_sets["val"], eval_sets["test"]],
            shuffle_label, feature_names, seed,
        )

    if time_aligned_K > 0:
        logger.info("Extracting time-aligned clips (K=%d per recording)...", time_aligned_K)
        X_train, Y_train = _extract_time_aligned(train_set, time_aligned_K, sfreq)
        X_val, Y_val = _extract_time_aligned(eval_sets["val"], time_aligned_K, sfreq)
        X_test, Y_test = _extract_time_aligned(eval_sets["test"], time_aligned_K, sfreq)
    else:
        logger.info("Extracting train features (%d recordings × %d passes)...",
                    len(train_set), n_passes)
        X_train, Y_train = _extract(
            train_set, n_passes, sfreq, seed,
            jepa=jepa_model, device=jepa_device, keep_channels=keep_channels,
        )
        logger.info("  X_train: %s, Y_train: %s", X_train.shape, Y_train.shape)

        logger.info("Extracting val features...")
        X_val_grp, Y_val_grp = _extract(
            eval_sets["val"], n_passes, sfreq, seed + 1, group_by_rec=True,
            jepa=jepa_model, device=jepa_device, keep_channels=keep_channels,
        )
        logger.info("  X_val:   %s, Y_val:   %s", X_val_grp.shape, Y_val_grp.shape)
        logger.info("Extracting test features...")
        X_test_grp, Y_test_grp = _extract(
            eval_sets["test"], n_passes, sfreq, seed + 2, group_by_rec=True,
            jepa=jepa_model, device=jepa_device, keep_channels=keep_channels,
        )
        logger.info("  X_test:  %s, Y_test:  %s", X_test_grp.shape, Y_test_grp.shape)
        # Flatten for Ridge fit/eval (n_rec * n_passes, ...)
        X_val = X_val_grp.reshape(-1, X_val_grp.shape[-1])
        Y_val = Y_val_grp.reshape(-1, Y_val_grp.shape[-1])
        X_test = X_test_grp.reshape(-1, X_test_grp.shape[-1])
        Y_test = Y_test_grp.reshape(-1, Y_test_grp.shape[-1])

    # Standardize features (helps ridge)
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True).clip(min=1e-8)
    X_train_n = (X_train - mu) / sd
    X_val_n = (X_val - mu) / sd
    X_test_n = (X_test - mu) / sd

    mode = []
    if shuffle_label:
        mode.append(f"shuffle={shuffle_label}")
    if time_aligned_K > 0:
        mode.append(f"time_aligned_K={time_aligned_K}")
    if not mode:
        mode.append(f"random_n_passes={n_passes}")
    print(f"\n=== Trivial-feature linear probe → movie features [{' + '.join(mode)}] ===")
    print(f"Setup: n_windows={n_windows}, ws={window_size_seconds}s, "
          f"norm_mode={norm_mode}, corrca={'yes' if corrca_filters else 'no'}")
    print(f"Features: {X_train.shape[1]} dims (per-channel mean+std + 5 bandpowers)")
    print(f"Train clips: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
    print()
    print(f"{'feature':<25} {'val corr':>10} {'test corr':>10}")
    print("-" * 50)
    metrics = {}
    feature_y_means = {}
    feature_y_stds = {}
    val_pred_norm = np.zeros((X_val_n.shape[0], len(feature_names)), dtype=np.float32)
    test_pred_norm = np.zeros((X_test_n.shape[0], len(feature_names)), dtype=np.float32)
    for i, fname_ in enumerate(feature_names):
        y_tr = Y_train[:, i]
        y_va = Y_val[:, i]
        y_te = Y_test[:, i]
        if np.std(y_tr) < 1e-10:
            print(f"{fname_:<25} {'(const)':>10} {'(const)':>10}")
            continue
        ym, ys = float(y_tr.mean()), float(y_tr.std())
        feature_y_means[fname_] = ym
        feature_y_stds[fname_] = ys
        probe = Ridge(alpha=1.0).fit(X_train_n, (y_tr - ym) / ys)
        pn_va = probe.predict(X_val_n)
        pn_te = probe.predict(X_test_n)
        val_pred_norm[:, i] = pn_va.astype(np.float32)
        test_pred_norm[:, i] = pn_te.astype(np.float32)
        pred_va = pn_va * ys + ym
        pred_te = pn_te * ys + ym
        c_va = pearsonr(pred_va, y_va).statistic if np.std(pred_va) > 0 else 0.0
        c_te = pearsonr(pred_te, y_te).statistic if np.std(pred_te) > 0 else 0.0
        print(f"{fname_:<25} {c_va:>+10.4f} {c_te:>+10.4f}")
        metrics[f"val/reg_{fname_}_corr"] = float(c_va)
        metrics[f"test/reg_{fname_}_corr"] = float(c_te)

    # Optional npz dump in the bootstrap_probe_eval.py schema, when we used
    # group_by_rec for eval. preds shape = (n_rec, n_passes, F) → matches the
    # expected (N_rec, T, F) layout the bootstrap helpers index along axis 0.
    if save_predictions_dir and not (shuffle_label or time_aligned_K > 0):
        save_dir = Path(save_predictions_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        n_features = len(feature_names)
        f_mean = np.array(
            [feature_y_means.get(f, 0.0) for f in feature_names], dtype=np.float32
        )
        f_std = np.array(
            [feature_y_stds.get(f, 1.0) for f in feature_names], dtype=np.float32
        )
        # NB: bootstrap unnormalizes preds via reg_preds * (f_std + 1e-8) + f_mean,
        # so we save the *normalized* predictions (matches probe_eval.py schema).
        for split, pn, Y_grp, X_grp in (
            ("val", val_pred_norm, Y_val_grp, X_val_grp),
            ("test", test_pred_norm, Y_test_grp, X_test_grp),
        ):
            n_rec, n_p, _ = X_grp.shape
            reg_preds = pn.reshape(n_rec, n_p, n_features)
            reg_targets = Y_grp.reshape(n_rec, n_p, n_features)
            # Compute median over training labels for cls compatibility
            f_median = np.array(
                [float(np.median(Y_train[:, i])) for i in range(n_features)],
                dtype=np.float32,
            )
            np.savez_compressed(
                save_dir / f"{split}_seed{seed}.npz",
                movie_reg_preds=reg_preds,
                movie_cls_logits=np.zeros_like(reg_preds, dtype=np.float32),
                movie_targets=reg_targets,
                feature_names=np.array(feature_names),
                feature_mean=f_mean,
                feature_std=f_std,
                feature_median=f_median,
                rec_ids=np.arange(n_rec),
                seed=seed,
            )
            logger.info("Wrote ridge predictions: %s", save_dir / f"{split}_seed{seed}.npz")

    if output_json:
        import json
        out = {
            "baseline": baseline,
            "seed": seed,
            "n_passes": n_passes,
            "n_chans": train_set.n_chans,
            "feat_dim": int(X_train.shape[1]),
            "metrics": metrics,
        }
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    fire.Fire(run)
