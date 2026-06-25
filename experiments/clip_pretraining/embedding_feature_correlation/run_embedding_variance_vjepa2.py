"""How dissimilar are V-JEPA-2 clip embeddings within vs across shots?

V-JEPA-2 emits one 1408-d embedding per 0.5 s clip (~2 Hz, ~406 clips for
"The_Present"). See `run_embedding_variance_clip.py` for the same analysis on
the per-frame 512-d OpenAI CLIP embeddings.

Goal: inform the EEG↔video contrastive pretraining objective. A shot-level
positive pair (two clips from the same shot, or clip↔shot-mean) is only useful
if same-shot embeddings are reliably *more* similar than different-shot ones.
This script quantifies that geometry in cosine space:

  1. Variance decomposition (eta^2): what fraction of embedding variance is
     *between* shots vs *within* shots. High eta^2 -> shot identity is a strong
     structuring factor -> shot-level positives are well separated from negatives.

  2. Pairwise cosine distributions: same-shot (candidate positives) vs
     different-shot (candidate negatives). Reports means, spread, and
     separability (d-prime, ROC-AUC of "is this a same-shot pair?").

  3. Temporal confound: same-shot pairs are also temporally adjacent. We bin all
     pairs by |time gap| and compare same- vs different-shot cosine *at matched
     gaps*, so the shot effect is not just "nearby in time".

  4. Per-shot compactness vs separation: cosine of each shot's clips to their own
     centroid vs to the nearest other shot centroid -> the per-shot margin a
     contrastive loss would have to exploit.

  5. Nearest-neighbor purity: for each clip, is its top-1 / top-5 neighbor in the
     same shot? This is how hard in-batch negatives are.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_embedding_variance_vjepa2.py

Outputs (alongside this file):
    vjepa2_embedding_variance_summary.txt
    vjepa2_embedding_variance_per_shot.csv
    vjepa2_embedding_variance.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import DEFAULT_EMBEDDINGS, DEFAULT_FEATURES  # noqa: E402
from run_shot_averaged_vjepa2 import assign_clip_shots  # noqa: E402

DEFAULT_OUTDIR = Path(__file__).resolve().parent


def variance_decomposition(Xn: np.ndarray, shot: np.ndarray) -> dict:
    """Between/within variance of L2-normalized embeddings, by shot.

    eta^2 = between_SS / total_SS in [0,1]: fraction of (angular) variance
    explained by shot membership. Also returns the F-like ratio.
    """
    mu = Xn.mean(axis=0)
    total_ss = float(((Xn - mu) ** 2).sum())
    shots = np.unique(shot)
    within_ss = 0.0
    between_ss = 0.0
    for s in shots:
        m = shot == s
        c = Xn[m].mean(axis=0)
        within_ss += float(((Xn[m] - c) ** 2).sum())
        between_ss += float(m.sum()) * float(((c - mu) ** 2).sum())
    n, k = len(Xn), len(shots)
    eta2 = between_ss / total_ss
    f_stat = (between_ss / (k - 1)) / (within_ss / (n - k)) if within_ss > 0 else float("inf")
    return {"eta2": eta2, "f_stat": f_stat, "within_ss": within_ss,
            "between_ss": between_ss, "total_ss": total_ss, "k_shots": k, "n_clips": n}


def pairwise_cosine_stats(Xn: np.ndarray, shot: np.ndarray, t: np.ndarray) -> dict:
    """Same-shot vs different-shot cosine over all i<j clip pairs."""
    S = Xn @ Xn.T
    iu, ju = np.triu_indices(len(Xn), k=1)
    cos = S[iu, ju]
    same = shot[iu] == shot[ju]
    gap = np.abs(t[iu] - t[ju])
    pos, neg = cos[same], cos[~same]
    dprime = (pos.mean() - neg.mean()) / np.sqrt(0.5 * (pos.var() + neg.var()))
    # ROC-AUC via Mann-Whitney: P(cos(pos) > cos(neg))
    order = np.argsort(cos)
    ranks = np.empty(len(cos)); ranks[order] = np.arange(1, len(cos) + 1)
    n_pos, n_neg = same.sum(), (~same).sum()
    auc = (ranks[same].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"cos": cos, "same": same, "gap": gap, "pos": pos, "neg": neg,
            "dprime": float(dprime), "auc": float(auc),
            "pos_mean": float(pos.mean()), "pos_std": float(pos.std()),
            "neg_mean": float(neg.mean()), "neg_std": float(neg.std()),
            "n_pos": int(n_pos), "n_neg": int(n_neg)}


def matched_gap_table(cos, same, gap, edges) -> pd.DataFrame:
    """Mean cosine for same- vs different-shot pairs within each |time gap| bin."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (gap >= lo) & (gap < hi)
        ps, ng = b & same, b & ~same
        rows.append({
            "gap_lo": lo, "gap_hi": hi,
            "same_mean": float(cos[ps].mean()) if ps.any() else np.nan,
            "same_n": int(ps.sum()),
            "diff_mean": float(cos[ng].mean()) if ng.any() else np.nan,
            "diff_n": int(ng.sum()),
        })
    return pd.DataFrame(rows)


def per_shot_geometry(Xn: np.ndarray, shot: np.ndarray) -> pd.DataFrame:
    """Per shot: within-shot compactness and separation to nearest other shot."""
    shots = np.unique(shot)
    cents = np.stack([Xn[shot == s].mean(axis=0) for s in shots])
    cents_n = cents / np.linalg.norm(cents, axis=1, keepdims=True)
    cc = cents_n @ cents_n.T
    np.fill_diagonal(cc, -np.inf)
    rows = []
    for idx, s in enumerate(shots):
        m = shot == s
        members = Xn[m]
        c = cents_n[idx]
        within = members @ c / np.linalg.norm(members, axis=1)  # members already unit
        nn = int(np.argmax(cc[idx]))
        rows.append({
            "shot_id": int(s), "n_clips": int(m.sum()),
            "within_cos_mean": float(within.mean()),
            "within_cos_min": float(within.min()),
            "nearest_other_shot": int(shots[nn]),
            "nearest_other_cos": float(cc[idx, nn]),
            "margin": float(within.mean() - cc[idx, nn]),
        })
    return pd.DataFrame(rows)


def nn_purity(Xn: np.ndarray, shot: np.ndarray, ks=(1, 5)) -> dict:
    """Fraction of clips whose top-k neighbors (excl. self) share its shot."""
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        same = (shot[topk] == shot[:, None])
        out[f"top{k}_purity"] = float(same.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.embeddings)
    X = data["embeddings"].astype(np.float64)
    t = data["timestamps"].astype(np.float64)
    freq = float(data["frequency"]) if "frequency" in data.files else 2.0
    feats = pd.read_parquet(args.features)
    clip_shot = assign_clip_shots(feats, t, freq=freq)

    keep = ~np.isnan(clip_shot)
    X, t, shot = X[keep], t[keep], clip_shot[keep].astype(int)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    print(f"clips={len(Xn)}, shots={len(np.unique(shot))}, dim={Xn.shape[1]}, "
          f"freq={freq} Hz")

    # Mean-centered + renormalized: removes the dominant shared component that
    # makes raw V-JEPA-2 cosines all sit near 1.0. This is the space a sensible
    # CLIP head operates in, so we report geometry here as the primary read and
    # keep raw cosine only for contrast.
    Xc = X - X.mean(axis=0, keepdims=True)
    Xcn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)

    vd = variance_decomposition(Xn, shot)
    pc_raw = pairwise_cosine_stats(Xn, shot, t)
    pc = pairwise_cosine_stats(Xcn, shot, t)
    edges = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    gap_tab = matched_gap_table(pc["cos"], pc["same"], pc["gap"], edges)
    geom = per_shot_geometry(Xcn, shot)
    purity = nn_purity(Xcn, shot)
    purity_raw = nn_purity(Xn, shot)

    # ---- summary text ----
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 70)
    emit("V-JEPA-2 clip embedding geometry: within-shot vs between-shot")
    emit("=" * 70)
    emit(f"\nclips={vd['n_clips']}  shots={vd['k_shots']}  cosine space (L2-normalized)")
    emit("\n[1] Variance decomposition (fraction of embedding variance):")
    emit(f"    between-shot (eta^2) = {vd['eta2']:.3f}   "
         f"within-shot = {1 - vd['eta2']:.3f}")
    emit(f"    F-statistic          = {vd['f_stat']:.1f}  "
         f"(>>1 means shot identity strongly structures the space)")
    emit("\n[2] Pairwise cosine similarity (RAW embeddings — anisotropic):")
    emit(f"    same-shot (positives) : {pc_raw['pos_mean']:.3f} +/- {pc_raw['pos_std']:.3f}  "
         f"(n={pc_raw['n_pos']:,})")
    emit(f"    diff-shot (negatives) : {pc_raw['neg_mean']:.3f} +/- {pc_raw['neg_std']:.3f}  "
         f"(n={pc_raw['n_neg']:,})")
    emit(f"    separation: d-prime = {pc_raw['dprime']:.2f}   ROC-AUC = {pc_raw['auc']:.3f}")
    emit("    -> all cosines crammed near 1.0: a dominant shared component "
         "dominates raw cosine.")
    emit("\n[2b] Pairwise cosine after MEAN-CENTERING (recommended CLIP space):")
    emit(f"    same-shot (positives) : {pc['pos_mean']:.3f} +/- {pc['pos_std']:.3f}")
    emit(f"    diff-shot (negatives) : {pc['neg_mean']:.3f} +/- {pc['neg_std']:.3f}")
    emit(f"    separation: d-prime = {pc['dprime']:.2f}   ROC-AUC = {pc['auc']:.3f}")
    emit("    (AUC = P[a same-shot pair is more similar than a different-shot pair])")
    emit("    -> centering opens up the dynamic range; same/diff become far more "
         "separable.")
    emit("\n[3] Same- vs different-shot cosine at matched |time gap| (s):")
    emit(f"    {'gap [s)':<12}{'same':>8}{'(n)':>9}{'diff':>8}{'(n)':>9}")
    for _, r in gap_tab.iterrows():
        same_s = f"{r['same_mean']:.3f}" if not np.isnan(r["same_mean"]) else "  -  "
        diff_s = f"{r['diff_mean']:.3f}" if not np.isnan(r["diff_mean"]) else "  -  "
        emit(f"    [{int(r['gap_lo']):>3},{int(r['gap_hi']):>4}){'':<3}{same_s:>8}"
             f"{int(r['same_n']):>9}{diff_s:>8}{int(r['diff_n']):>9}")
    emit("    -> compare same vs diff WITHIN a row: gap above the shot effect alone.")
    emit("\n[4] Per-shot geometry (within-shot compactness vs nearest other shot):")
    emit(f"    within-shot cos to own centroid : "
         f"{geom['within_cos_mean'].mean():.3f} (mean over shots)")
    emit(f"    cos to nearest OTHER centroid   : "
         f"{geom['nearest_other_cos'].mean():.3f}")
    emit(f"    margin (own - nearest other)    : {geom['margin'].mean():.3f}  "
         f"[min {geom['margin'].min():.3f}, "
         f"{int((geom['margin'] < 0).sum())} shots overlap a neighbor]")
    emit("\n[4,5 computed in mean-centered space.]")
    emit("\n[5] Nearest-neighbor shot purity (in-batch negative hardness):")
    emit(f"    top-1 neighbor same shot : {purity['top1_purity']:.3f}  "
         f"(raw: {purity_raw['top1_purity']:.3f})")
    emit(f"    top-5 neighbor same shot : {purity['top5_purity']:.3f}  "
         f"(raw: {purity_raw['top5_purity']:.3f})")

    emit("\n" + "-" * 70)
    emit("CLIP design implications:")
    emit(f"  * CENTER/WHITEN BEFORE CONTRASTING. Raw cosine AUC={pc_raw['auc']:.2f} "
         f"with everything")
    emit(f"    near 1.0; after mean-centering AUC={pc['auc']:.2f}. The discriminative")
    emit("    signal lives in a small off-mean component, so either subtract the")
    emit("    global mean / batch-norm the targets, or rely on a learnable")
    emit("    temperature small enough to expand that range.")
    if pc["auc"] > 0.85 and vd["eta2"] > 0.4:
        emit(f"  * Strong shot structure (eta^2={vd['eta2']:.2f}, centered AUC="
             f"{pc['auc']:.2f}): same-shot")
        emit("    clips are reliably closer than cross-shot ones. Shot-level positives")
        emit("    (clip<->shot-mean, or two clips from one shot) are well-posed and")
        emit("    cross-shot pairs are mostly clean negatives.")
    else:
        emit("  * Shot structure is moderate: same/diff overlap -> every cross-shot")
        emit("    pair as a hard negative will inject label noise.")
    emit("  * Temporally adjacent clips (gap<~2s) are very similar even ACROSS shot")
    emit("    boundaries (see [3]) -> exclude near-in-time clips from the negative")
    emit("    pool (false negatives), and beware that 'same-shot positive' is partly")
    emit("    just 'nearby in time'.")
    emit(f"  * top-1 NN purity {purity['top1_purity']:.2f}: ~"
         f"{int(round((1-purity['top1_purity'])*100))}% of clips' nearest neighbor "
         f"is in ANOTHER shot")
    emit("    -> in-batch negatives include genuinely hard (near-duplicate) cases;")
    emit("    a moderate temperature + many negatives will help more than mining.")

    summ = args.outdir / "vjepa2_embedding_variance_summary.txt"
    summ.write_text("\n".join(lines) + "\n")
    geom_csv = args.outdir / "vjepa2_embedding_variance_per_shot.csv"
    geom.to_csv(geom_csv, index=False)
    print(f"\nSaved {summ}")
    print(f"Saved {geom_csv}")

    # ---- plots ----
    # Top row gets 3 panels so the raw and centered cosine histograms each have
    # their own x-range — overlaying them is hopeless because raw cosines are
    # crammed into 0.97-1.0 (huge density spike) while centered span -1..1.
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_cent = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[0, 2])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_margin = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[1, 2]); ax_text.axis("off")

    for axh, p, title in [
        (ax_raw, pc_raw,
         f"RAW cosine — d'={pc_raw['dprime']:.2f}, AUC={pc_raw['auc']:.2f}"),
        (ax_cent, pc,
         f"MEAN-CENTERED — d'={pc['dprime']:.2f}, AUC={pc['auc']:.2f}"),
    ]:
        lo = min(p["neg"].min(), p["pos"].min())
        bins = np.linspace(lo, 1.0, 60)
        axh.hist(p["neg"], bins=bins, density=True, alpha=0.6, color="crimson",
                 label=f"diff-shot  {p['neg_mean']:.3f}±{p['neg_std']:.3f}")
        axh.hist(p["pos"], bins=bins, density=True, alpha=0.6, color="steelblue",
                 label=f"same-shot  {p['pos_mean']:.3f}±{p['pos_std']:.3f}")
        axh.axvline(p["neg_mean"], color="crimson", ls="--", lw=1)
        axh.axvline(p["pos_mean"], color="steelblue", ls="--", lw=1)
        axh.set_xlabel("cosine similarity"); axh.set_ylabel("density")
        axh.set_title(title, fontsize=10)
        axh.legend(fontsize=8, loc="upper left")

    ax = ax_gap
    g = gap_tab.dropna(subset=["same_mean"])
    mid = (g["gap_lo"] + g["gap_hi"]) / 2
    ax.plot(mid, g["same_mean"], "o-", color="steelblue", label="same-shot")
    gd = gap_tab.dropna(subset=["diff_mean"])
    midd = (gd["gap_lo"] + gd["gap_hi"]) / 2
    ax.plot(midd, gd["diff_mean"], "s-", color="crimson", label="different-shot")
    ax.set_xscale("symlog")
    ax.set_xlabel("|time gap| between clips (s)"); ax.set_ylabel("mean cosine (centered)")
    ax.set_title("Cosine vs time gap (temporal confound)")
    ax.legend(fontsize=8)

    ax = ax_scatter
    ax.scatter(geom["within_cos_mean"], geom["nearest_other_cos"],
               s=20 + geom["n_clips"], c=geom["margin"], cmap="coolwarm_r",
               edgecolor="k", linewidth=0.3)
    lim = [min(geom["within_cos_mean"].min(), geom["nearest_other_cos"].min()) - 0.02,
           1.0]
    ax.plot(lim, lim, "k--", lw=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("within-shot cos to own centroid (centered)")
    ax.set_ylabel("cos to nearest OTHER centroid")
    ax.set_title("Per-shot compactness vs separation\n(below diagonal = separable)")

    ax = ax_margin
    ax.hist(geom["margin"], bins=20, color="seagreen", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(geom["margin"].mean(), color="darkgreen", ls="--",
               label=f"mean {geom['margin'].mean():.3f}")
    ax.set_xlabel("per-shot margin (own - nearest other centroid cos)")
    ax.set_ylabel("# shots")
    ax.set_title("Contrastive margin per shot")
    ax.legend(fontsize=8)

    summary = (
        f"$\\bf{{Headline}}$\n"
        f"eta^2 = {vd['eta2']:.3f}   F = {vd['f_stat']:.1f}\n"
        f"({vd['k_shots']} shots, {vd['n_clips']} clips)\n\n"
        f"$\\bf{{Cosine\\ separability}}$\n"
        f"raw:       d' = {pc_raw['dprime']:.2f}   AUC = {pc_raw['auc']:.3f}\n"
        f"centered: d' = {pc['dprime']:.2f}   AUC = {pc['auc']:.3f}\n\n"
        f"$\\bf{{NN\\ purity\\ (centered)}}$\n"
        f"top-1: {purity['top1_purity']:.3f}\n"
        f"top-5: {purity['top5_purity']:.3f}\n\n"
        f"$\\bf{{Overlapping\\ shots}}$\n"
        f"{int((geom['margin'] < 0).sum())}/{len(geom)} have negative margin"
    )
    ax_text.text(0.0, 1.0, summary, ha="left", va="top", fontsize=10,
                 family="monospace", transform=ax_text.transAxes)

    fig.suptitle(f"V-JEPA-2 embedding geometry for CLIP design  "
                 f"(eta^2={vd['eta2']:.2f}, F={vd['f_stat']:.0f})", fontsize=13)
    fig.tight_layout()
    out_png = args.outdir / "vjepa2_embedding_variance.png"
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
