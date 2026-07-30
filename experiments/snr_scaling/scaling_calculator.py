"""Design calculator for cross-subject SNR aggregation on stimulus-locked EEG.

Answers: how far can averaging K subjects at the same movie moment take a
stimulus-feature probe, and where does HBN sit in the (anchors x subjects)
design space relative to other naturalistic-EEG datasets?

Model
-----
Neural response of subject s at movie moment t:

    x_s(t) = g(t) + n_s(t)          Var(g) = sg2,  Var(n) = sn2

  g   shared stimulus response (identical across subjects by definition)
  n   subject fingerprint + noise (independent across subjects)

Single-trial reliability (== inter-subject correlation at this time
resolution, in whatever multivariate space the probe reads):

    rho1 = sg2 / (sg2 + sn2)

Averaging K independent subjects at the same t leaves g untouched and shrinks
the noise variance by K, so reliability follows Spearman-Brown:

    R(K) = K*rho1 / (1 + (K-1)*rho1)

A frozen V-JEPA-2 feature y is a deterministic function of the movie, so it
carries no measurement noise. The attainable correlation is therefore

    r(K) <= sqrt(R(K)) * corr(g, y)          with corr(g, y) <= 1

so sqrt(R(K)) is a hard ceiling on any probe, for any encoder or objective.

Run:  uv run --group eeg python experiments/snr_scaling/scaling_calculator.py
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Measured quantities from this repo ------------------------------------
# RESULTS_jul7.md 3.4 / 4.3, R6 test, mean Pearson r over 12 movie features.
R_OBS = {
    "random init": 0.0527,
    "fresh500 scene_clip": 0.1328,
    "vanilla CLIP 400ep": 0.1459,
    "soft-target tau=0.05 400ep": 0.1517,
    "REVE warm-start + scene_clip": 0.1715,
}
BEST_FROM_SCRATCH = 0.1517
BEST_OVERALL = 0.1715

# hbn.py MOVIE_METADATA; anchors = distinct non-overlapping 2 s movie moments.
WINDOW_S = 2.0
HBN_MOVIES = {"ThePresent": 203.3, "DespicableMe": 170.6}

# Recording counts implied exactly by the retrieval / probe JSONs
# (29593 = 293 x 101 val, 10908 = 108 x 101 test, ~71k = ~703 x 101 train).
HBN_RECORDINGS = {"train R1-R4": 703, "val R5": 293, "test R6": 108}


def reliability(rho1: float, k: int) -> float:
    """Spearman-Brown reliability of a K-subject average."""
    return k * rho1 / (1.0 + (k - 1) * rho1)


def ceiling(rho1: float, k: int) -> float:
    """Upper bound on |corr(K-subject average, noiseless feature)|."""
    return math.sqrt(reliability(rho1, k))


def implied_rho1(r_observed: float, corr_g_y: float = 1.0) -> float:
    """Smallest rho1 consistent with an observed single-trial probe r.

    corr_g_y = 1 assumes the feature is perfectly encoded in the noiseless
    neural response, which is optimistic -- so this is a LOWER bound on rho1.
    """
    return (r_observed / corr_g_y) ** 2


def main() -> None:
    ks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    rhos = [0.02, 0.05, 0.10, 0.28]

    print("=" * 74)
    print("1. CEILING ON PROBE r WHEN AVERAGING K SUBJECTS AT THE SAME MOMENT")
    print("=" * 74)
    print("rho1 = single-trial inter-subject reliability (2 s resolution).")
    print("0.02-0.05 broadband; 0.10-0.28 delta/theta (experiments.md:47-70).\n")
    header = "  rho1  " + "".join(f"{('K=' + str(k)):>8}" for k in ks)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rho in rhos:
        row = f"  {rho:.2f}  " + "".join(f"{ceiling(rho, k):8.3f}" for k in ks)
        print(row)

    print("\n" + "=" * 74)
    print("2. WHERE THE MEASURED RESULTS SIT (all at K=1, single trial)")
    print("=" * 74)
    for name, r in R_OBS.items():
        print(f"  {name:<30} r = {r:.4f}   implies rho1 >= {implied_rho1(r):.4f}")

    print("\n  Headroom at K=1 (single-trial testing), best from-scratch"
          f" r = {BEST_FROM_SCRATCH:.4f}:")
    for rho in rhos:
        c = ceiling(rho, 1)
        print(f"    if rho1 = {rho:.2f}: ceiling {c:.3f}"
              f"  -> achieved {100 * BEST_FROM_SCRATCH / c:5.1f}% of it"
              f"  (max further gain {c / BEST_FROM_SCRATCH:.2f}x)")

    print("\n" + "=" * 74)
    print("3. THE TWO AXES BUY DIFFERENT THINGS")
    print("=" * 74)
    print("  K_test  = subjects averaged at evaluation -> RAISES the ceiling")
    print("  K_train = partners in the pretraining target -> approaches it\n")
    rho = 0.05
    print(f"  At rho1 = {rho:.2f}:")
    for k in [1, 2, 5, 10, 20, 50]:
        c = ceiling(rho, k)
        print(f"    K_test = {k:>3}   ceiling r = {c:.3f}"
              f"   ({c / ceiling(rho, 1):.2f}x the single-trial ceiling)")
    print("\n  vs. K_train, which cannot exceed the K_test=1 ceiling"
          f" of {ceiling(rho, 1):.3f}")
    print("  -> objective work is capped at a ~1.5-2x gain;"
          " test-time aggregation is 4-6x.")

    print("\n" + "=" * 74)
    print("4. THE OTHER AXIS: HOW MANY DISTINCT STIMULUS ANCHORS EXIST")
    print("=" * 74)
    total_anchors = 0
    for movie, dur in HBN_MOVIES.items():
        a = int(dur // WINDOW_S)
        total_anchors += a
        print(f"  {movie:<16} {dur:6.1f} s  ->  {a:4d} distinct 2 s anchors")
    print(f"  {'TOTAL (wired)':<16} {sum(HBN_MOVIES.values()):6.1f} s"
          f"  ->  {total_anchors:4d} anchors")
    print("\n  Recordings per anchor (= subjects available to average):")
    for split, n in HBN_RECORDINGS.items():
        print(f"    {split:<14} {n:4d}")
    print("\n  NOTE: 71k train windows sounds large, but it is"
          f" {HBN_RECORDINGS['train R1-R4']} subjects x 101 moments.")
    print("  More subjects buys SNR per anchor. It buys ZERO new anchors.")

    print("\n" + "=" * 74)
    print("5. DESIGN SPACE: anchors (A) x subjects per anchor (S)")
    print("=" * 74)
    print("  Approximate; verify before citing.\n")
    datasets = [
        ("HBN TP (this work)", 101, 703, "video, naturalistic"),
        ("HBN TP+DM", 186, 703, "video, naturalistic"),
        ("HBN 4-clip battery (if wired)", 350, 700, "video, naturalistic"),
        ("THINGS-EEG2", 16740, 10, "static images"),
        ("DEAP", 1200, 32, "music video"),
        ("Broderick Natural Speech", 1800, 19, "audio only"),
    ]
    print(f"  {'dataset':<32}{'A':>7}{'S':>7}{'A*S':>10}   stimulus")
    print("  " + "-" * 70)
    for name, a, s, kind in datasets:
        print(f"  {name:<32}{a:>7}{s:>7}{a * s:>10}   {kind}")
    print("\n  HBN and THINGS-EEG2 sit at OPPOSITE corners.")
    print("  Nobody occupies the high-A x high-S corner --"
          " that is the gap the paper names.")

    _plot(rhos, datasets)


def _plot(rhos, datasets) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ks = list(range(1, 201))
    for rho in rhos:
        ax1.plot(ks, [ceiling(rho, k) for k in ks], label=f"ISC $\\rho_1$={rho:.2f}")
    ax1.axhline(BEST_FROM_SCRATCH, ls="--", c="k", lw=1)
    ax1.text(2, BEST_FROM_SCRATCH + 0.015,
             f"best from-scratch, K=1 ({BEST_FROM_SCRATCH:.3f})", fontsize=8)
    ax1.axhline(BEST_OVERALL, ls=":", c="k", lw=1)
    ax1.text(2, BEST_OVERALL + 0.015,
             f"REVE warm-start, K=1 ({BEST_OVERALL:.3f})", fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("K = subjects averaged at the same movie moment")
    ax1.set_ylabel("ceiling on probe Pearson $r$")
    ax1.set_title("Cross-subject averaging raises the ceiling")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for name, a, s, _ in datasets:
        ax2.scatter(a, s, s=45)
        ax2.annotate(name, (a, s), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("A = distinct stimulus anchors")
    ax2.set_ylabel("S = subjects per anchor")
    ax2.set_title("Naturalistic EEG design space (the empty corner)")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"experiments/snr_scaling/snr_scaling.{ext}", dpi=200)
    print("\nWrote experiments/snr_scaling/snr_scaling.{png,pdf}")


if __name__ == "__main__":
    main()
