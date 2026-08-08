"""Convert a pretrained REVE checkpoint to an ``encoder_init_from`` artifact.

REVE's state_dict matches EEGEncoderTokens' state_dict 1:1 except for:
  - to_patch_embedding.{0.weight,0.bias} (REVE has an extra `.0.`)
  - final_layer.* classification head (drop entirely)
  - channel_positions / cls_query_token buffers, which EET recomputes itself

This script produces a `.pth.tar` with `model_state_dict` keys prefixed
`encoder.` so the trainer's existing `--meta.encoder_init_from=<path>` flag
loads them directly. No trainer code change needed.

Two source modes:

``--hf-repo`` calls ``REVE.from_pretrained``. Note that ``brain-bzh/reve-base``
is a **gated** repo: it needs an authenticated HF token, and it 401s otherwise.

``--local-snapshot`` reads ``weights.safetensors`` straight out of an already
downloaded HF snapshot directory. The published snapshot ships ``model_cfg.yaml``
and ``weights.safetensors``, not the ``config.json`` / ``model.safetensors`` pair
``from_pretrained`` expects, so this is the only path that works offline or
without gate access. Its keys already carry the ``encoder.`` prefix.

Usage:
    PYTHONPATH=. .venv/bin/python \\
        experiments/clip_pretraining/scene_clip_from_checkpoint/prepare_reve_checkpoint.py \\
        --local-snapshot ~/.cache/huggingface/hub/models--eeg-telecom-paris--reve-base/snapshots/<sha> \\
        --output reve_base_eet_init.pth.tar
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from braindecode.models import REVE
from safetensors.torch import load_file

# EET recomputes channel coordinates from chs_info and has no attention-pooling
# query, so these REVE buffers have nowhere to land. They must be dropped here:
# anything left over arrives at load_encoder_weights as an unexpected key, which
# the strict-load guard treats as a shape/topology mismatch and rejects.
NON_ENCODER_KEYS = frozenset(
    {"channel_positions", "cls_query_token", "_position_bank.embedding"}
)


def remap_reve_to_eet(reve_sd: dict) -> dict:
    """Convert REVE.state_dict() key naming to EEGEncoderTokens, prefixed `encoder.`."""
    out = {}
    for k, v in reve_sd.items():
        # Drop classification head — we only want the encoder.
        if k.startswith("final_layer.") or k in NON_ENCODER_KEYS:
            continue
        # Rename the patch-embedding submodule.
        k2 = re.sub(r"^to_patch_embedding\.0\.", "to_patch_embedding.", k)
        out[f"encoder.{k2}"] = v
    return out


def remap_safetensors_to_eet(sd: dict) -> dict:
    """Convert published weights.safetensors keys, which are already `encoder.`-prefixed."""
    out = {}
    for k, v in sd.items():
        # The published file carries channel_positions / cls_query_token unprefixed,
        # so filtering on the prefix already excludes them.
        if not k.startswith("encoder."):
            continue
        k2 = re.sub(
            r"^encoder\.to_patch_embedding\.0\.", "encoder.to_patch_embedding.", k
        )
        out[k2] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--hf-repo",
        help="HuggingFace repo id. brain-bzh/reve-base is gated and needs a token.",
    )
    src.add_argument(
        "--local-snapshot",
        help="Path to a downloaded HF snapshot dir containing weights.safetensors",
    )
    ap.add_argument("--n-chans", type=int, default=129)
    ap.add_argument("--n-times", type=int, default=400)
    ap.add_argument("--sfreq", type=int, default=200)
    ap.add_argument(
        "--output", required=True, help="Where to save the converted .pth.tar"
    )
    args = ap.parse_args()

    if args.local_snapshot:
        weights = Path(args.local_snapshot) / "weights.safetensors"
        if not weights.is_file():
            raise SystemExit(f"No weights.safetensors under {args.local_snapshot}")
        print(f"Loading {weights}...")
        raw_sd = load_file(str(weights))
        print(f"  safetensors: {len(raw_sd)} keys")
        new_sd = remap_safetensors_to_eet(raw_sd)
        source = f"local-snapshot:{args.local_snapshot}"
    else:
        print(f"Loading {args.hf_repo} (downloads on first run)...")
        reve = REVE.from_pretrained(
            args.hf_repo,
            n_outputs=10,  # placeholder; final_layer is dropped anyway
            n_chans=args.n_chans,
            n_times=args.n_times,
            sfreq=args.sfreq,
        )
        raw_sd = reve.state_dict()
        print(f"  REVE state_dict: {len(raw_sd)} keys")
        new_sd = remap_reve_to_eet(raw_sd)
        source = args.hf_repo

    dropped = sorted(set(raw_sd) - {k for k in raw_sd if k.startswith("encoder.")})
    print(
        f"  → EET-compatible: {len(new_sd)} keys "
        f"({sum(v.numel() for v in new_sd.values()):,} params), dropped {dropped}"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": new_sd,
            "epoch": -1,
            "step": 0,
            "source": source,
            "format": "encoder.<eet-key>",
        },
        out_path,
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print()
    print("Use with the trainer via:")
    print(f"  --meta.encoder_init_from={out_path.resolve()}")


if __name__ == "__main__":
    main()
