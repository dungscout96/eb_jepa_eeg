"""Extract autoresearch summary block from a run.log.

Reads the trailing `---` summary printed by experiments/eeg_jepa/main.py and
emits a JSON dict with all fields, plus the precomputed `val_corr_weighted`.

Usage
-----
    python autoresearch/parse_log.py path/to/run.log
    python autoresearch/parse_log.py path/to/run.log --tsv-row <commit>
"""

import argparse
import json
import re
import sys
from pathlib import Path

EXPECTED_KEYS = [
    "val_corr_weighted",
    "val_corr_weighted_max",
    "val_corr_weighted_max_ep",
    "val_reg_position",
    "val_reg_contrast",
    "val_reg_luminance",
    "val_reg_narrative",
    "sanity_linear_probe_acc",
    "sanity_emb_var_mean",
    "sanity_emb_var_min",
    "sanity_cosim_mean",
    "sanity_cosim_max",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "num_params_M",
    "num_steps",
]

# Hard collapse gates — runs failing either are auto-discarded regardless of
# val_corr_weighted_max. Thresholds picked from cross-run analysis (see
# autoresearch/analysis/metric_correlation_report.md §6 "Fallback / sanity floor").
COLLAPSE_VAR_MIN_FLOOR = 0.01    # sanity/embedding_variance_min must exceed this
COLLAPSE_COSIM_MAX_CEIL = 0.99   # sanity/cosim_random_pairs_max must be below this


def collapse_gate(parsed: dict) -> tuple[bool, str]:
    """Return (passed, reason). Reason is empty when passed."""
    var_min = parsed.get("sanity_emb_var_min", 0.0)
    cosim_max = parsed.get("sanity_cosim_max", 1.0)
    if var_min <= COLLAPSE_VAR_MIN_FLOOR:
        return False, f"emb_var_min={var_min:.4f}<={COLLAPSE_VAR_MIN_FLOOR}"
    if cosim_max >= COLLAPSE_COSIM_MAX_CEIL:
        return False, f"cosim_max={cosim_max:.4f}>={COLLAPSE_COSIM_MAX_CEIL}"
    return True, ""


def parse(log_path: Path) -> dict | None:
    text = log_path.read_text()

    # The summary block is the LAST '---' line followed by `key: value` lines
    # (matches main.py's `print("---")` block).
    chunks = text.split("\n---\n")
    if len(chunks) < 2:
        # Try `--- ` as first/last line of file (no leading newline).
        idx = text.rfind("---")
        if idx == -1:
            return None
        tail = text[idx:]
    else:
        tail = "---\n" + chunks[-1]

    out = {}
    for line in tail.splitlines():
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s+([-+0-9.eE]+)\s*$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        try:
            out[key] = float(val) if "." in val or "e" in val.lower() else int(val)
        except ValueError:
            continue

    if not all(k in out for k in EXPECTED_KEYS):
        missing = [k for k in EXPECTED_KEYS if k not in out]
        out["_missing_keys"] = missing
        return out if any(k in out for k in EXPECTED_KEYS) else None

    return out


def tsv_row(commit: str, parsed: dict) -> str:
    """Format a results.tsv row from parsed summary.

    Schema (17 columns):
        commit  val_corr_weighted  val_corr_weighted_max  val_corr_weighted_max_ep
        val_reg_position  val_reg_contrast  val_reg_luminance  val_reg_narrative
        sanity_lin_probe_acc  sanity_emb_var_mean  sanity_emb_var_min
        sanity_cosim_mean  sanity_cosim_max
        peak_vram_gb  collapse_gate  status  description
    """
    if parsed is None or "_missing_keys" in parsed:
        empty = "0.000000\t" * 11 + "0.000000\t-1\t"
        return f"{commit}\t{empty}0.0\tn/a\tcrash\t"

    peak_vram_gb = parsed["peak_vram_mb"] / 1024.0
    gated, gate_reason = collapse_gate(parsed)
    gate_str = "ok" if gated else f"fail:{gate_reason}"
    status = "keep" if gated else "discard"

    return (
        f"{commit}\t"
        f"{parsed['val_corr_weighted']:.6f}\t"
        f"{parsed['val_corr_weighted_max']:.6f}\t"
        f"{int(parsed['val_corr_weighted_max_ep'])}\t"
        f"{parsed['val_reg_position']:.6f}\t"
        f"{parsed['val_reg_contrast']:.6f}\t"
        f"{parsed['val_reg_luminance']:.6f}\t"
        f"{parsed['val_reg_narrative']:.6f}\t"
        f"{parsed['sanity_linear_probe_acc']:.6f}\t"
        f"{parsed['sanity_emb_var_mean']:.6f}\t"
        f"{parsed['sanity_emb_var_min']:.6f}\t"
        f"{parsed['sanity_cosim_mean']:.6f}\t"
        f"{parsed['sanity_cosim_max']:.6f}\t"
        f"{peak_vram_gb:.1f}\t"
        f"{gate_str}\t"
        f"{status}\t"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_path", type=Path)
    p.add_argument("--tsv-row", metavar="COMMIT", help="emit a results.tsv row instead of JSON")
    args = p.parse_args()

    parsed = parse(args.log_path)
    if parsed is None:
        print("ERROR: no parseable summary block in", args.log_path, file=sys.stderr)
        sys.exit(1)

    if args.tsv_row:
        print(tsv_row(args.tsv_row, parsed))
    else:
        print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
