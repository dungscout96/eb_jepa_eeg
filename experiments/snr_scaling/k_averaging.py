"""E0.2 -- empirical K-subject averaging curve, and a direct test of Spearman-Brown.

E0.1 measured ``rho1`` and *extrapolated* a ceiling with Spearman-Brown:

    R(K) = K*rho1 / (1 + (K-1)*rho1),   ceiling(K) = sqrt(R(K))

Every claim about test-time subject aggregation -- the ~3x lever that now
carries more of the argument than single-trial work does -- rests on that
extrapolation being right. This script stops assuming it and measures it.

Three things are produced, in increasing order of what they cost to get.

**1. Empirical reliability R(K), encoder-free.** Draw two DISJOINT sets of K
subjects, average each set's embeddings per movie anchor, and correlate the two
averages. That correlation estimates R(K) directly, with no model and no probe
in the loop. Overlaying it on the predicted curve is the actual validation of
Spearman-Brown on this data. If these disagree, the ceiling numbers in
RESULTS.md are wrong and nothing downstream survives.

**2. Probe r vs K, embedding-space.** Encode each subject, average K subjects'
embeddings per anchor, then apply the ridge fit on the train split. This is
what a deployed system that pools several recordings would actually do.

**3. Probe r vs K, signal-space.** Average the raw (per-recording z-scored) EEG
across K subjects first, then encode once. The classical ERP logic.

**The gap between 2 and 3 is itself a result.** If the encoder were linear in
the noise the two curves would coincide. Divergence says the nonlinearity
matters, and which one is higher says whether to aggregate before or after the
encoder -- a concrete deployment recommendation.

Protocol note -- why r(K=1) here is NOT the r in RESULTS_jul7
--------------------------------------------------------------
``probe_traintest.py`` computes Pearson r over all ~11 K (subject, anchor)
pairs, where each of the 101 feature values recurs once per subject. Here, r is
computed over the **101 anchor-level points** after averaging K subjects, for
every K including K=1. That is the only way the curve is internally comparable
across K, and it is the quantity ``sqrt(R(K))`` actually bounds. Expect
r(K=1) here to differ from 0.1517; what matters is the SHAPE against the
predicted curve, and the K=1 value is reported so the offset is explicit.

Per-subject normalisation is applied BEFORE averaging in signal space, matching
``norm_mode=per_recording`` at training time. Averaging un-normalised recordings
would let a high-amplitude subject dominate -- the same scale-sensitivity that
separated raw Sahani-Linden from ISC in ``noise_ceiling.py``.

Usage
-----
    PYTHONPATH=. uv run --group eeg python experiments/snr_scaling/k_averaging.py \
        --checkpoint /path/latest.pth.tar --config /path/config_TP.yaml \
        --eval-split test --rho1-json experiments/snr_scaling/isc_test_ThePresent.json \
        --output experiments/snr_scaling/k_averaging_test.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from eb_jepa.evaluation.clip_probe.probe import (
    SCALAR_FEATURES_DEFAULT,
    build_dataset,
    embed_recording_all_windows,
    load_encoder_state,
)
from eb_jepa.datasets.hbn import _read_raw_windows
from eb_jepa.logging import get_logger
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import load_config

logger = get_logger(__name__)

DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_anchors(dataset):
    """Movie-time anchors shared by every recording, plus per-recording indices.

    Joins on ``t_start_recordings`` (movie seconds), the same key paired.py and
    retrieval.py use -- not window index.
    """
    keyed = [
        {int(round(float(t) * 1000)): i for i, t in enumerate(ts)}
        for ts in dataset.t_start_recordings
    ]
    anchors = sorted(set.intersection(*(set(k) for k in keyed)))
    if not anchors:
        raise RuntimeError("No movie time is shared by all recordings.")
    idx = np.stack([[k[a] for a in anchors] for k in keyed])  # [R, A]
    subjects = [
        str((m or {}).get("subject", (m or {}).get("participant_id", f"__rec{i}")))
        for i, m in enumerate(dataset._recording_metadata)
    ]
    return anchors, idx, subjects


def embed_eval_aligned(encoder, dataset, anchor_idx, device, batch_size):
    """Encode every recording, keeping [R, A, D] alignment. Also returns Y [A, F].

    Feature values are identical across recordings at the same anchor (they are
    a function of the movie), so Y is taken from recording 0 and verified.
    """
    n_rec, n_anchors = anchor_idx.shape
    X = None
    y_ref = None
    for r in range(n_rec):
        X_rec, Y_rec = embed_recording_all_windows(
            encoder, dataset, r, device, batch_size)
        sel = anchor_idx[r]
        if X is None:
            X = np.empty((n_rec, n_anchors, X_rec.shape[1]), dtype=np.float32)
            y_ref = Y_rec[sel]
        X[r] = X_rec[sel]
        if (r + 1) % 25 == 0:
            logger.info("  embedded %d/%d recordings", r + 1, n_rec)
    return X, y_ref


def load_eval_signals(dataset, anchor_idx):
    """Per-recording z-scored raw EEG at the shared anchors -> [R, A, C, T].

    Normalisation matches embed_recording_all_windows so signal-space and
    embedding-space differ ONLY in where the averaging happens.

    NOTE: all channels are kept, including the flat EGI 'Cz' reference. The
    encoder was trained on 129 channels and must be fed 129; dropping it here
    (as measure_isc.py correctly does for ISC) would break the encoder.
    """
    n_rec, n_anchors = anchor_idx.shape
    out = None
    for r in range(n_rec):
        raw = _read_raw_windows(dataset._fif_paths[r], dataset._crop_inds[r])
        eeg = torch.from_numpy(raw)
        if dataset._norm_mode == "per_recording":
            mu = eeg.mean(dim=(0, 2), keepdim=True)
            sd = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
            eeg = (eeg - mu) / sd
        else:
            eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
        sel = anchor_idx[r]
        if out is None:
            out = np.empty((n_rec, n_anchors) + tuple(eeg.shape[1:]), dtype=np.float32)
        out[r] = eeg.numpy()[sel]
        if (r + 1) % 25 == 0:
            logger.info("  loaded signals %d/%d recordings", r + 1, n_rec)
    return out


def encode_windows(encoder, eeg, device, batch_size):
    """Encode [A, C, T] -> [A, D] with the same path as the probe."""
    x = torch.from_numpy(eeg).unsqueeze(1)          # [A, 1, C, T]
    outs = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            b = x[s:s + batch_size].to(device)
            tokens = encoder.encode_tokens(b, mask=None)
            pooled = encoder.pool_to_windows(tokens)
            outs.append(pooled.squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# The three measurements
# ---------------------------------------------------------------------------

def empirical_reliability(X, ks, n_draws, rng):
    """Measure R(K) directly: correlation of two DISJOINT K-subject averages.

    Encoder-free in the sense that no probe is involved -- this is a property of
    the representation, compared against the Spearman-Brown prediction.
    Correlation is computed per embedding dimension across anchors and averaged,
    weighting dimensions equally.

    Needs 2K <= n_recordings, so large K are skipped rather than silently
    computed on overlapping sets.
    """
    n_rec = X.shape[0]
    out = {}
    for k in ks:
        if 2 * k > n_rec:
            continue
        vals = []
        for _ in range(n_draws):
            perm = rng.permutation(n_rec)
            a = X[perm[:k]].mean(axis=0)              # [A, D]
            b = X[perm[k:2 * k]].mean(axis=0)
            a = a - a.mean(axis=0, keepdims=True)
            b = b - b.mean(axis=0, keepdims=True)
            na = np.linalg.norm(a, axis=0)
            nb = np.linalg.norm(b, axis=0)
            good = (na > 0) & (nb > 0)
            if not good.any():
                continue
            vals.append(float(((a[:, good] * b[:, good]).sum(axis=0)
                               / (na[good] * nb[good])).mean()))
        if vals:
            out[str(k)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_draws": len(vals),
            }
    return out


def probe_curve(get_embedding, X_shape_r, ks, n_draws, rng, models, scaler,
                Y, feature_names):
    """Probe r vs K for one aggregation mode.

    ``get_embedding(subject_indices) -> [A, D]`` abstracts over embedding-space
    (cheap: mean of precomputed embeddings) and signal-space (expensive: average
    raw EEG then run the encoder).
    """
    out = {}
    for k in ks:
        if k > X_shape_r:
            continue
        # With k == n_recordings there is exactly one possible draw.
        draws = 1 if k == X_shape_r else n_draws
        per_draw = []
        for _ in range(draws):
            sel = rng.choice(X_shape_r, size=k, replace=False)
            emb = get_embedding(sel)                      # [A, D]
            emb_n = scaler.transform(emb)
            rs = []
            for i, feat in enumerate(feature_names):
                reg = models.get(feat)
                if reg is None:
                    continue
                y = Y[:, i]
                valid = ~np.isnan(y)
                if valid.sum() < 10 or y[valid].std() < 1e-9:
                    continue
                pred = reg.predict(emb_n[valid])
                if pred.std() < 1e-12:
                    continue
                rs.append(float(pearsonr(pred, y[valid]).statistic))
            if rs:
                per_draw.append(float(np.mean(rs)))
        if per_draw:
            out[str(k)] = {
                "mean_r": float(np.mean(per_draw)),
                "std_r": float(np.std(per_draw)),
                "n_draws": len(per_draw),
            }
    return out


def predicted_ceiling(rho1, ks):
    def c(k):
        if not math.isfinite(rho1) or rho1 <= 0:
            return float("nan")
        return math.sqrt(k * rho1 / (1.0 + (k - 1) * rho1))
    return {str(k): c(k) for k in ks}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"])
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--max-train-recordings", type=int, default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--random-baseline", action="store_true")
    ap.add_argument("--skip-signal-space", action="store_true",
                    help="Embedding-space only. Signal space needs the dense "
                         "[R, A, C, T] array (~2.3 GB for R6).")
    ap.add_argument("--rho1-json", default=None,
                    help="isc_*.json from E0.1; its corrca_held_out_combined "
                         "sets the predicted curve.")
    ap.add_argument("--rho1", type=float, default=None,
                    help="Override rho1 directly instead of reading the JSON.")
    ap.add_argument("--output", default="experiments/snr_scaling/k_averaging.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    logger.info("Building TRAIN split ...")
    train_set = build_dataset(cfg, "train", SCALAR_FEATURES_DEFAULT)
    logger.info("Building %s split ...", args.eval_split.upper())
    eval_set = build_dataset(cfg, args.eval_split, SCALAR_FEATURES_DEFAULT)

    encoder = build_encoder(
        cfg, n_chans=train_set.n_chans, n_times=train_set.n_times,
        chs_info=train_set.get_chs_info(), n_windows=cfg.data.n_windows,
    )
    if not args.random_baseline:
        assert args.checkpoint, "Provide --checkpoint or --random-baseline"
        load_encoder_state(encoder, args.checkpoint)
    encoder = encoder.to(device).eval()

    # --- fit the ridge on train, exactly as probe_traintest does -----------
    logger.info("Embedding TRAIN ...")
    n_tr = len(train_set) if args.max_train_recordings is None else args.max_train_recordings
    Xs, Ys = [], []
    for r in range(min(n_tr, len(train_set))):
        xr, yr = embed_recording_all_windows(encoder, train_set, r, device,
                                             args.encode_batch)
        Xs.append(xr)
        Ys.append(yr)
        if (r + 1) % 50 == 0:
            logger.info("  %d/%d train recordings", r + 1, n_tr)
    X_tr = np.concatenate(Xs, axis=0)
    Y_tr = np.concatenate(Ys, axis=0)
    del Xs, Ys
    scaler = StandardScaler().fit(X_tr)
    X_tr_n = scaler.transform(X_tr)

    models = {}
    for i, feat in enumerate(SCALAR_FEATURES_DEFAULT):
        y = Y_tr[:, i]
        valid = ~np.isnan(y)
        if valid.sum() < 100 or y[valid].std() < 1e-9:
            continue
        reg = RidgeCV(alphas=np.logspace(-2, 4, 13))
        reg.fit(X_tr_n[valid], y[valid])
        models[feat] = reg
    logger.info("Fitted %d ridge heads on %d train windows", len(models), len(X_tr))
    del X_tr, X_tr_n, Y_tr

    # --- aligned eval embeddings ------------------------------------------
    anchors, anchor_idx, subjects = align_anchors(eval_set)
    n_rec = anchor_idx.shape[0]
    logger.info("%s: %d recordings, %d unique subjects, %d shared anchors",
                args.eval_split.upper(), n_rec, len(set(subjects)), len(anchors))

    X_ev, Y_anchor = embed_eval_aligned(encoder, eval_set, anchor_idx, device,
                                        args.encode_batch)
    logger.info("Eval embeddings %s", X_ev.shape)

    ks = [k for k in args.ks if k <= n_rec]

    logger.info("Empirical reliability R(K) via disjoint halves ...")
    emp_rel = empirical_reliability(X_ev, ks, args.n_draws, rng)

    logger.info("Probe curve, embedding-space ...")
    emb_curve = probe_curve(
        lambda sel: X_ev[sel].mean(axis=0),
        n_rec, ks, args.n_draws, rng, models, scaler, Y_anchor,
        SCALAR_FEATURES_DEFAULT)

    sig_curve = {}
    if not args.skip_signal_space:
        logger.info("Loading raw signals for signal-space averaging ...")
        sig = load_eval_signals(eval_set, anchor_idx)
        logger.info("Signals %s (%.2f GB)", sig.shape, sig.nbytes / 1e9)
        logger.info("Probe curve, signal-space ...")
        sig_curve = probe_curve(
            lambda sel: encode_windows(encoder, sig[sel].mean(axis=0), device,
                                       args.encode_batch),
            n_rec, ks, args.n_draws, rng, models, scaler, Y_anchor,
            SCALAR_FEATURES_DEFAULT)
        del sig

    # --- predicted curve from E0.1 ----------------------------------------
    rho1 = args.rho1
    rho1_source = "cli"
    if rho1 is None and args.rho1_json:
        d = json.loads(Path(args.rho1_json).read_text())
        rho1 = d["rho1_candidates"].get("corrca_held_out_combined")
        rho1_source = f"{args.rho1_json}:corrca_held_out_combined"
    pred = predicted_ceiling(rho1, ks) if rho1 else {}

    result = {
        "eval_split": args.eval_split,
        "checkpoint": args.checkpoint,
        "random_baseline": args.random_baseline,
        "n_recordings": int(n_rec),
        "n_unique_subjects": len(set(subjects)),
        "n_anchors": len(anchors),
        "n_draws": args.n_draws,
        "ks": ks,
        "rho1": rho1,
        "rho1_source": rho1_source,
        "predicted_ceiling": pred,
        "empirical_reliability": emp_rel,
        "probe_r_embedding_space": emb_curve,
        "probe_r_signal_space": sig_curve,
        "n_ridge_heads": len(models),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))

    # --- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"E0.2 K-averaging -- {args.eval_split} "
          f"({n_rec} recordings, {len(anchors)} anchors, {args.n_draws} draws)")
    print("=" * 78)
    print(f"{'K':>5}{'R(K) pred':>12}{'R(K) meas':>12}{'ceil pred':>12}"
          f"{'r embed':>10}{'r signal':>10}")
    print("-" * 78)
    for k in ks:
        sk = str(k)
        rp = (k * rho1 / (1 + (k - 1) * rho1)) if rho1 else float("nan")
        rm = emp_rel.get(sk, {}).get("mean", float("nan"))
        cp = float(pred.get(sk, float("nan"))) if pred else float("nan")
        re_ = emb_curve.get(sk, {}).get("mean_r", float("nan"))
        rs = sig_curve.get(sk, {}).get("mean_r", float("nan"))
        print(f"{k:>5}{rp:>12.4f}{rm:>12.4f}{cp:>12.3f}{re_:>10.3f}{rs:>10.3f}")
    print("\n  R(K) pred vs meas is the Spearman-Brown validation.")
    print("  r embed vs r signal: divergence means the encoder is not linear")
    print("  in the noise; the higher one is where to aggregate in deployment.")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
