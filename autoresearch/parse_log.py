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
    "val_reg_position",
    "val_reg_contrast",
    "val_reg_luminance",
    "val_reg_narrative",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "num_params_M",
    "num_steps",
]


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
    """Format a results.tsv row from parsed summary."""
    if parsed is None or "_missing_keys" in parsed:
        return f"{commit}\t0.000000\t0.000000\t0.000000\t0.000000\t0.000000\t0.0\tcrash\t"
    peak_vram_gb = parsed["peak_vram_mb"] / 1024.0
    return (
        f"{commit}\t"
        f"{parsed['val_corr_weighted']:.6f}\t"
        f"{parsed['val_reg_position']:.6f}\t"
        f"{parsed['val_reg_contrast']:.6f}\t"
        f"{parsed['val_reg_luminance']:.6f}\t"
        f"{parsed['val_reg_narrative']:.6f}\t"
        f"{peak_vram_gb:.1f}\t"
        f"keep\t"
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
