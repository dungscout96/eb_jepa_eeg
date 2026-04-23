"""Raw-EEG data analysis: ISC + variance decomposition.

Measures the data-level stimulus-locked ceiling, independent of any encoder.

For each val recording, extract K time-aligned clips (evenly spaced across the
movie). Build tensor Z[S, K, C, T] where S=recordings, K=clips, C=channels,
T=samples. Compute:

  1. Inter-Subject Correlation (ISC) per channel on raw 129-ch and CorrCA-5
     EEG. Pooled ISC = stimulus-locked variance ceiling (under the linear
     subject-averaging view).

  2. Per-band ISC (delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-80)
     via bandpass → ISC. Tells us where stimulus signal lives in frequency.

  3. Variance decomposition of raw EEG values:
       Var_total   = per-channel total variance across (S, K, T)
       Var_subject = variance explained by subject mean
       Var_stim    = variance explained by time-bin mean (across subjects)
     Report η²_subj and η²_stim per channel + pooled.

  4. Read out CorrCA eigenvalues from corrca_filters.npz — these ARE the
     top-5 ISC of CorrCA components.

Outputs (saved to --out_dir):
  - data.npz          raw arrays used for computation (optional, may be large)
  - report.json
  - report.md
  - isc_per_channel.png
  - variance_per_channel.png
"""

import json
from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.training_utils import load_config
from experiments.eeg_jepa.main import resolve_preprocessed_dir


BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
}


def bandpass(x, fs, lo, hi, order=4):
    """Zero-phase bandpass filter on last axis."""
    nyq = fs / 2
    sos = butter(order, [lo / nyq, min(hi, 0.95 * nyq) / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def extract_aligned_clips(dataset, n_time_bins=20, max_recordings=None):
    """Extract time-aligned clips from each recording.

    Returns
    -------
    eeg      : np.ndarray [S, K, C, T] (raw 129-ch EEG after per-rec z-norm)
    corrca   : np.ndarray [S, K, Kc, T] or None (CorrCA-projected EEG)
    rec_idx  : np.ndarray [S]
    """
    n_rec = len(dataset) if max_recordings is None else min(max_recordings, len(dataset))
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
    sfreq = dataset.sfreq

    all_eeg, all_corrca, rec_ids = [], [], []

    for rec_idx in range(n_rec):
        crop_inds = dataset._crop_inds[rec_idx]
        n_clips_total = len(crop_inds) - required + 1
        if n_clips_total <= 0:
            continue

        # n_time_bins evenly spaced starts across the recording
        starts = np.linspace(0, n_clips_total - 1, n_time_bins, dtype=int)

        rec_clips_raw, rec_clips_cca = [], []
        for start in starts:
            window_idx = list(range(start, start + required, dataset.temporal_stride))
            eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[window_idx])
            # eeg_np: [n_windows, C, T]
            eeg_np = eeg_np.reshape(-1, eeg_np.shape[-1])  # concat windows → [C, n_win*T]
            # per-rec z-normalization (same as training)
            eeg = torch.from_numpy(eeg_np.copy())
            m = eeg.mean(dim=-1, keepdim=True)
            s = eeg.std(dim=-1, keepdim=True).clamp(min=1e-8)
            eeg = (eeg - m) / s
            rec_clips_raw.append(eeg.numpy())

            if dataset._corrca_W is not None:
                # project 129 → k
                cca = torch.einsum("ct,ck->kt", eeg, dataset._corrca_W)
                rec_clips_cca.append(cca.numpy())

        all_eeg.append(np.stack(rec_clips_raw))      # [K, C, T]
        if dataset._corrca_W is not None:
            all_corrca.append(np.stack(rec_clips_cca))
        rec_ids.append(rec_idx)

        if (rec_idx + 1) % 20 == 0:
            print(f"  loaded {rec_idx + 1}/{n_rec} recordings")

    eeg = np.stack(all_eeg).astype(np.float32)       # [S, K, C, T]
    corrca = np.stack(all_corrca).astype(np.float32) if all_corrca else None
    return eeg, corrca, np.array(rec_ids), sfreq


def isc_per_channel(Z):
    """ISC per channel using N*var(mean) - 1)/(N-1) formula.

    Z: [S, K, C, T] — already per-subject z-normalized on the raw axis.
    For each (k, c): stack [S, T] → z-score per subject over T,
    mean across S, compute var(mean) over T, convert to ISC.

    Returns isc [C]  (averaged over time bins k).
    """
    S, K, C, T = Z.shape
    iscs = np.zeros(C, dtype=np.float64)
    for k in range(K):
        # Zk: [S, C, T]
        Zk = Z[:, k]
        # z-score per (subject, channel) over T
        Zk = (Zk - Zk.mean(axis=-1, keepdims=True)) / (Zk.std(axis=-1, keepdims=True) + 1e-8)
        mean_ts = Zk.mean(axis=0)          # [C, T]
        var_mean = mean_ts.var(axis=-1)    # [C]
        isc_k = (S * var_mean - 1) / (S - 1)
        iscs += isc_k
    return iscs / K


def isc_per_band(Z, sfreq):
    """ISC per channel, per frequency band.

    Returns dict band → isc [C].
    """
    out = {}
    for name, (lo, hi) in BANDS.items():
        try:
            Zb = bandpass(Z.reshape(-1, Z.shape[-1]), sfreq, lo, hi).reshape(Z.shape).astype(np.float32)
            out[name] = isc_per_channel(Zb)
        except Exception as e:
            print(f"  band {name} failed: {e}")
    return out


def variance_decomposition_raw(Z):
    """Per-channel η²_subject vs η²_stim on raw EEG values.

    Z: [S, K, C, T] — already z-normed per subject.
    Treating (K, T) together as "within-subject variability", we flatten to
    [S, C, M] where M = K*T.
    """
    S, K, C, T = Z.shape
    Zf = Z.transpose(0, 2, 1, 3).reshape(S, C, K * T)  # [S, C, M]

    # per-channel total variance
    var_total = Zf.var(axis=(0, 2))                    # [C]

    # subject mean
    subj_mean = Zf.mean(axis=2)                        # [S, C]
    var_subj = subj_mean.var(axis=0)                   # [C]

    # time-bin mean (stimulus): within each time bin k, mean across subjects.
    # Reshape back to [S, C, K, T] then mean over S.
    Zf2 = Zf.reshape(S, C, K, T)
    stim_mean = Zf2.mean(axis=0)                       # [C, K, T]
    # variance of stim_mean across (K, T) = stimulus-locked variance fraction
    var_stim = stim_mean.var(axis=(1, 2))              # [C]

    eta2_subj = var_subj / (var_total + 1e-12)
    eta2_stim = var_stim / (var_total + 1e-12)
    return eta2_subj, eta2_stim, var_total


def run(
    config: str = "experiments/eeg_jepa/cfgs/default.yaml",
    out_dir: str = "raw_data_analysis",
    n_time_bins: int = 20,
    max_recordings: int | None = None,
    norm_mode: str = "per_recording",
    corrca_filters: str | None = "corrca_filters.npz",
    preprocessed_dir: str | None = None,
    save_arrays: bool = False,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    overrides = {"data.norm_mode": norm_mode}
    if corrca_filters:
        overrides["data.corrca_filters"] = corrca_filters
    cfg = load_config(config, overrides)

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
    print(f"  {len(val_set)} val recordings, {val_set.n_chans} channels (post-CorrCA if enabled)")

    print(f"Extracting {n_time_bins} aligned clips per recording...")
    eeg, corrca, rec_ids, sfreq = extract_aligned_clips(
        val_set, n_time_bins=n_time_bins, max_recordings=max_recordings,
    )
    S, K, C, T = eeg.shape
    print(f"  raw EEG shape: {eeg.shape}, sfreq={sfreq}")
    if corrca is not None:
        print(f"  CorrCA shape: {corrca.shape}")

    if save_arrays:
        np.savez(out / "data.npz", eeg=eeg, corrca=corrca if corrca is not None else np.array([]),
                 rec_ids=rec_ids, sfreq=sfreq)

    # ---- ISC raw 129 ----
    print("Computing ISC per channel (broadband)...")
    isc_raw = isc_per_channel(eeg)
    print(f"  broadband ISC: mean={isc_raw.mean():.4f}, median={np.median(isc_raw):.4f}, "
          f"top-5 ch={np.sort(isc_raw)[-5:].round(4).tolist()}")

    # ---- ISC per band ----
    print("Computing ISC per band...")
    isc_band_raw = isc_per_band(eeg, sfreq)
    for name, v in isc_band_raw.items():
        print(f"  band {name}: mean ISC={v.mean():.4f}, top-5={np.sort(v)[-5:].round(4).tolist()}")

    # ---- CorrCA components ----
    isc_corrca = None
    if corrca is not None:
        print("Computing ISC on CorrCA components...")
        isc_corrca = isc_per_channel(corrca)
        print(f"  CorrCA ISC per component: {isc_corrca.round(4).tolist()}")

    # ---- CorrCA eigenvalues from file ----
    corrca_eig = None
    if corrca_filters and Path(corrca_filters).exists():
        cc = np.load(corrca_filters)
        corrca_eig = cc["isc_values"].tolist()
        print(f"  CorrCA saved eigenvalues (train-fit): {[round(v, 4) for v in corrca_eig]}")

    # ---- Variance decomposition raw 129 ----
    print("Variance decomposition (raw EEG)...")
    eta2_subj, eta2_stim, vtot = variance_decomposition_raw(eeg)
    print(f"  raw 129-ch:  mean η²_subj={eta2_subj.mean():.4f}, "
          f"mean η²_stim={eta2_stim.mean():.4f}, "
          f"pooled stim/(stim+subj)={eta2_stim.sum()/(eta2_stim.sum()+eta2_subj.sum()+1e-12):.4f}")

    # ---- Variance decomposition CorrCA ----
    eta_cca_subj, eta_cca_stim = None, None
    if corrca is not None:
        eta_cca_subj, eta_cca_stim, _ = variance_decomposition_raw(corrca)
        print(f"  CorrCA 5-ch: η²_subj={eta_cca_subj.round(4).tolist()}, "
              f"η²_stim={eta_cca_stim.round(4).tolist()}")

    # ---- Report ----
    report = {
        "n_recordings": int(S),
        "n_time_bins": int(K),
        "n_channels_raw": int(C),
        "samples_per_clip": int(T),
        "sfreq": float(sfreq),
        "isc_raw_mean": float(isc_raw.mean()),
        "isc_raw_median": float(np.median(isc_raw)),
        "isc_raw_top10": np.sort(isc_raw)[-10:].tolist(),
        "isc_band_mean": {k: float(v.mean()) for k, v in isc_band_raw.items()},
        "isc_band_max_per_ch": {k: float(v.max()) for k, v in isc_band_raw.items()},
        "isc_corrca_components": isc_corrca.tolist() if isc_corrca is not None else None,
        "corrca_train_eigenvalues": corrca_eig,
        "eta2_subj_raw_mean": float(eta2_subj.mean()),
        "eta2_stim_raw_mean": float(eta2_stim.mean()),
        "eta2_subj_corrca": eta_cca_subj.tolist() if eta_cca_subj is not None else None,
        "eta2_stim_corrca": eta_cca_stim.tolist() if eta_cca_stim is not None else None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(isc_raw, bins=40, edgecolor="black")
    axes[0].axvline(isc_raw.mean(), color="red", ls="--", label=f"mean={isc_raw.mean():.3f}")
    axes[0].set_title("Broadband ISC per raw channel")
    axes[0].set_xlabel("ISC")
    axes[0].legend()

    x = np.arange(C)
    axes[1].bar(x, eta2_subj, alpha=0.5, label="η²_subj")
    axes[1].bar(x, eta2_stim, alpha=0.7, label="η²_stim", color="red")
    axes[1].set_title("Variance attribution per raw channel")
    axes[1].set_xlabel("channel"); axes[1].set_ylabel("η²")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out / "isc_per_channel.png", dpi=150)

    if isc_corrca is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.bar(np.arange(len(isc_corrca)), isc_corrca, label="val ISC")
        if corrca_eig is not None:
            ax.bar(np.arange(len(corrca_eig)) + 0.35, corrca_eig, width=0.35, label="train eig", alpha=0.7)
        ax.set_title("CorrCA components — ISC per component")
        ax.set_xlabel("component"); ax.set_ylabel("ISC")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out / "corrca_isc.png", dpi=150)

    # ---- Markdown summary ----
    md = ["# Raw-EEG data analysis\n"]
    md.append(f"- {S} val recordings × {K} time-aligned clips × {C} channels × {T} samples @ {sfreq}Hz\n\n")
    md.append("## Inter-Subject Correlation (raw 129 ch, broadband)\n")
    md.append(f"- Mean ISC per channel: **{isc_raw.mean():.4f}**\n")
    md.append(f"- Median ISC per channel: {np.median(isc_raw):.4f}\n")
    md.append(f"- Top-10 channels ISC: {[round(v, 4) for v in np.sort(isc_raw)[-10:]]}\n\n")
    md.append("## ISC per band (mean over channels)\n")
    md.append("|band|mean ISC|max ISC (best ch)|\n|--|--|--|\n")
    for k, v in isc_band_raw.items():
        md.append(f"|{k}|{v.mean():.4f}|{v.max():.4f}|\n")
    md.append("\n")

    if isc_corrca is not None:
        md.append("## CorrCA components\n")
        md.append(f"- Per-component ISC (val split): {[round(v, 4) for v in isc_corrca]}\n")
        if corrca_eig is not None:
            md.append(f"- CorrCA train eigenvalues (from npz): {[round(v, 4) for v in corrca_eig]}\n")
        md.append("\n")
    md.append("## Variance decomposition (raw 129 ch)\n")
    md.append(f"- Mean η²_subj = **{eta2_subj.mean():.4f}**\n")
    md.append(f"- Mean η²_stim = **{eta2_stim.mean():.4f}**\n")
    md.append(f"- Stim / (stim+subj) pooled: {eta2_stim.sum()/(eta2_stim.sum()+eta2_subj.sum()+1e-12):.4f}\n\n")
    if eta_cca_subj is not None:
        md.append("## Variance decomposition (CorrCA components)\n")
        md.append(f"- η²_subj per component: {[round(v, 4) for v in eta_cca_subj]}\n")
        md.append(f"- η²_stim per component: {[round(v, 4) for v in eta_cca_stim]}\n\n")

    md.append("## Interpretation\n")
    md.append("- Mean ISC = the fraction of EEG that is stimulus-locked across subjects, on average per channel. 0 = all subject noise, 1 = every subject sees identical signal.\n")
    md.append("- η²_stim on raw = ceiling any encoder can reach for stimulus representation, under a linear subject-averaging view. Compare to embedding η²_stim ≈ 0.005-0.008 in Exp 6.\n")
    md.append("- CorrCA eigenvalues are the top-k ISC-maximizing projections, so they ARE the upper bound for any 5-dim linear stimulus representation.\n")
    (out / "report.md").write_text("".join(md))
    print(f"\nAll outputs saved to: {out}")


if __name__ == "__main__":
    fire.Fire(run)
