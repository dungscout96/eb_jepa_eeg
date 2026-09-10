"""Download CBraMod's released weights and convert them to an encoder_init_from artifact.

Counterpart of ``prepare_reve_checkpoint.py`` for the HBN-free warm start.
``weighting666/CBraMod/pretrained_weights.pth`` is a bare state_dict of the
upstream ``CBraMod`` module; it maps onto
:class:`eb_jepa.cbramod.CBraModEncoderTokens` by two renames and one drop:

  - ``encoder.layers.*``  -> ``transformer.layers.*``   (the criss-cross stack)
  - ``patch_embedding.*`` -> unchanged
  - ``proj_out.*``        -> dropped (pretraining output projection; the
                              upstream fine-tuning heads replace it with
                              Identity, and so does this port)

Every surviving key is then prefixed ``encoder.`` so the trainer's existing
``--meta.encoder_init_from=<path>`` loads it through the strict guard in
``load_encoder_weights`` with zero missing and zero unexpected keys. The script
asserts exactly that against a freshly built encoder before writing anything,
so a silent partial load cannot leave here as an artifact.

Usage:
    PYTHONPATH=. uv run --group eeg python \\
        experiments/clip_pretraining/scene_clip_from_checkpoint/prepare_cbramod_checkpoint.py \\
        --output /work/hdd/bbnv/kkokate/eb_jepa/cbramod_eet_init.pth.tar
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from eb_jepa.cbramod import CBraModEncoderTokens
from eb_jepa.training_utils import load_encoder_weights

HF_REPO = "weighting666/CBraMod"
HF_FILE = "pretrained_weights.pth"


def remap_cbramod_to_eet(sd: dict) -> dict:
    """Upstream CBraMod state_dict -> ``encoder.<CBraModEncoderTokens key>``."""
    out = {}
    for k, v in sd.items():
        if k.startswith("proj_out."):
            continue
        k2 = re.sub(r"^encoder\.layers\.", "transformer.layers.", k)
        out[f"encoder.{k2}"] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--hf-file", default=HF_FILE)
    ap.add_argument(
        "--local-weights",
        default=None,
        help="Skip the download and read this pretrained_weights.pth.",
    )
    ap.add_argument("--n-chans", type=int, default=129)
    ap.add_argument("--n-times", type=int, default=400)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = args.local_weights or hf_hub_download(args.hf_repo, args.hf_file)
    print(f"Reading {src}")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    print(f"  upstream state_dict: {len(sd)} keys")

    new_sd = remap_cbramod_to_eet(sd)
    print(
        f"  -> {len(new_sd)} encoder keys (dropped {len(sd) - len(new_sd)} proj_out keys)"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": new_sd,
        "epoch": -1,
        "step": 0,
        "source": f"{args.hf_repo}/{args.hf_file}" if not args.local_weights else src,
        "format": "encoder.<cbramod-key>",
    }
    torch.save(payload, out_path)

    # Round-trip through the real loader at the sweep's geometry. The
    # channel count is not a parameter shape in CBraMod, so this holds for any
    # montage, but the sweep's own numbers are the ones worth asserting on.
    enc = CBraModEncoderTokens(n_chans=args.n_chans, n_times=args.n_times, n_windows=1)
    info = load_encoder_weights(enc, out_path)
    assert not info["missing"] and not info["unexpected"], info
    print(
        f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB); "
        f"strict load OK ({info['n_loaded']} tensors)"
    )
    print(f"  --meta.encoder_init_from={out_path.resolve()}")


if __name__ == "__main__":
    main()
