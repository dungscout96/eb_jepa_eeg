"""Download brain-bzh/reve-base and convert to an encoder_init_from artifact.

REVE's state_dict matches EEGEncoderTokens' state_dict 1:1 except for:
  - to_patch_embedding.{0.weight,0.bias} (REVE has an extra `.0.`)
  - final_layer.* classification head (drop entirely)

This script produces a `.pth.tar` with `model_state_dict` keys prefixed
`encoder.` so the trainer's existing `--meta.encoder_init_from=<path>` flag
loads them directly. No trainer code change needed.

Usage:
    PYTHONPATH=. .venv/bin/python \\
        experiments/clip_pretraining/scene_clip_from_checkpoint/prepare_reve_checkpoint.py \\
        --output reve_base_eet_init.pth.tar
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from braindecode.models import REVE


def remap_reve_to_eet(reve_sd: dict) -> dict:
    """Convert REVE's state_dict to EEGEncoderTokens key naming, prefixed `encoder.`."""
    out = {}
    for k, v in reve_sd.items():
        # Drop classification head — we only want the encoder.
        if k.startswith("final_layer."):
            continue
        # Rename the patch-embedding submodule.
        k2 = re.sub(r"^to_patch_embedding\.0\.", "to_patch_embedding.", k)
        out[f"encoder.{k2}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="brain-bzh/reve-base",
                    help="HuggingFace repo id for the pretrained REVE checkpoint")
    ap.add_argument("--n-chans", type=int, default=129)
    ap.add_argument("--n-times", type=int, default=400)
    ap.add_argument("--sfreq", type=int, default=200)
    ap.add_argument("--output", required=True,
                    help="Where to save the converted .pth.tar")
    args = ap.parse_args()

    print(f"Loading {args.hf_repo} (downloads on first run)...")
    reve = REVE.from_pretrained(
        args.hf_repo,
        n_outputs=10,  # placeholder; final_layer is dropped anyway
        n_chans=args.n_chans,
        n_times=args.n_times,
        sfreq=args.sfreq,
    )
    reve_sd = reve.state_dict()
    print(f"  REVE state_dict: {len(reve_sd)} keys")

    new_sd = remap_reve_to_eet(reve_sd)
    print(f"  → EET-compatible: {len(new_sd)} keys (dropped {len(reve_sd) - len(new_sd)} non-encoder keys)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": new_sd, "epoch": -1, "step": 0,
         "source": args.hf_repo, "format": "encoder.<eet-key>"},
        out_path,
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print()
    print("Use with the trainer via:")
    print(f"  --meta.encoder_init_from={out_path.resolve()}")


if __name__ == "__main__":
    main()
