#!/usr/bin/env python3
"""Estimate pretraining memory for OpenEEGBench backbones on HBN data.

For each backbone listed at
https://github.com/braindecode/OpenEEGBench/blob/main/docs/backbones.md, this
script:

  1. Loads the model via ``open_eeg_bench.default_configs.backbones.<name>()``
     (downloads pretrained weights from HuggingFace on first run).
  2. Reports parameter count and static training memory:
       - fp32 weights         (4 * N bytes)
       - fp32 gradients       (4 * N bytes)
       - AdamW optimizer state (8 * N bytes: first & second moments)
       Total static = 16 * N bytes for pure-fp32 AdamW.
  3. Builds an HBN-style input batch (B, C, T) — derived from
     ``eb_jepa/training/cfgs/default.yaml`` and ``eb_jepa/datasets/hbn.py``:
       n_chans=129, sfreq=200Hz, window_size=2s, n_windows=4
       → flat per-sample tensor (129, 800) when n_windows are concatenated,
         or (4, 129, 400) as the dataset emits.
  4. Runs forward+backward (or forward-only) for each requested batch size,
     measuring activation memory analytically with forward hooks (works on
     CPU). On CUDA, also reports torch.cuda.max_memory_allocated.
  5. Prints a per-model table and saves a CSV + PNG plot of total memory vs
     batch size.

Usage:
    uv run scripts/estimate_memory.py
    uv run scripts/estimate_memory.py --models biot,reve --batch-sizes 1,8,32
    uv run scripts/estimate_memory.py --device cuda --measure-backward
    uv run scripts/estimate_memory.py --no-download   # skip HF weight download
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Constants derived from eb_jepa HBN config
# ----------------------------------------------------------------------------

HBN_N_CHANS = 129
HBN_SFREQ = 200
HBN_WINDOW_SECONDS = 2
HBN_N_WINDOWS = 4
HBN_FEATURES = 4  # contrast_rms, luminance_mean, position_in_movie, narrative_event_score

HBN_N_TIMES_PER_WINDOW = int(HBN_SFREQ * HBN_WINDOW_SECONDS)            # 400
HBN_N_TIMES_FLAT = HBN_N_WINDOWS * HBN_N_TIMES_PER_WINDOW                # 1600

DTYPE_BYTES = 4  # fp32

DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# OpenEEGBench backbones list (from docs/backbones.md). Native input shapes
# from each model's pretraining recipe are used as a fallback if the HBN-shaped
# tensor breaks the backbone (e.g. fixed-channel models without an interpolator).
BACKBONES = [
    # (factory_name, native_chans, native_times, native_sfreq, notes)
    ("biot",        16,  2000, 200, "Yang+ 2023 — fixed 16ch; uses Interpolated wrapper for 129ch"),
    ("labram",      64,  200,  200, "Jiang+ 2024 — uses InterpolatedLaBraM"),
    ("bendr",       20,  1536, 256, "Kostas+ 2021 — uses InterpolatedBENDR"),
    ("cbramod",     19,  6000, 200, "Wang+ 2025"),
    ("signal_jepa", 8,   1024, 256, "Guetschel+ 2024 — uses InterpolatedSignalJEPA"),
    ("reve",        128, 1024, 256, "Music+ 2025"),
    # ("eegpt",     58,  1024, 256, "uses InterpolatedEEGPT"),  # not in OEB 0.6.0 yet

]


# ----------------------------------------------------------------------------
# Activation memory probe
# ----------------------------------------------------------------------------


@dataclass
class ActivationProbe:
    """Sum the bytes of every forward-output tensor as a proxy for activation memory."""
    total_bytes: int = 0
    per_module: dict[str, int] = field(default_factory=dict)
    _handles: list = field(default_factory=list)

    def _hook(self, name):
        def fn(_module, _inp, out):
            n = 0
            if isinstance(out, torch.Tensor):
                n = out.numel() * out.element_size()
            elif isinstance(out, (list, tuple)):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        n += o.numel() * o.element_size()
            self.total_bytes += n
            self.per_module[name] = self.per_module.get(name, 0) + n
        return fn

    def attach(self, model: nn.Module):
        for name, m in model.named_modules():
            if len(list(m.children())) == 0:  # leaf modules only
                self._handles.append(m.register_forward_hook(self._hook(name)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:6.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def static_memory(n_params: int) -> dict[str, int]:
    """Per-parameter fp32 AdamW cost breakdown (bytes)."""
    w = n_params * DTYPE_BYTES
    g = n_params * DTYPE_BYTES
    adam = n_params * DTYPE_BYTES * 2  # m + v
    return {
        "weights": w,
        "gradients": g,
        "adam_state": adam,
        "total_static": w + g + adam,
    }


def make_input(batch_size: int, chans: int, times: int, device: str) -> torch.Tensor:
    return torch.randn(batch_size, chans, times, device=device)


def try_forward(
    model: nn.Module,
    batch_size: int,
    chans: int,
    times: int,
    device: str,
    measure_backward: bool,
) -> tuple[int, int | None]:
    """Run forward (+ optional backward); return (activation_bytes, cuda_peak_bytes_or_None).

    activation_bytes is from a hook-based sum of leaf-module outputs; this
    approximates the tensors that backprop must keep alive.
    cuda_peak_bytes is the real allocator high-water mark if device==cuda.
    """
    model.eval() if not measure_backward else model.train()
    x = make_input(batch_size, chans, times, device)

    probe = ActivationProbe()
    probe.attach(model)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    try:
        if measure_backward:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(f"Unexpected output type: {type(out)}")
            loss = out.float().pow(2).mean()
            loss.backward()
        else:
            with torch.no_grad():
                _ = model(x)
    finally:
        probe.detach()

    cuda_peak = None
    if device == "cuda":
        torch.cuda.synchronize()
        cuda_peak = torch.cuda.max_memory_allocated()

    return probe.total_bytes, cuda_peak


def _build_chs_info():
    """GSN-HydroCel-129 channel info, matching HBN dataset."""
    import mne
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    info = mne.create_info(montage.ch_names, sfreq=HBN_SFREQ, ch_types="eeg")
    import numpy as np
    raw = mne.io.RawArray(np.zeros((len(montage.ch_names), 1)), info, verbose=False)
    raw.set_montage(montage, verbose=False)
    return raw.info["chs"]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="all",
                   help=f"Comma-separated subset of {[b[0] for b in BACKBONES]}, or 'all'.")
    p.add_argument("--batch-sizes", default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
                   help="Comma-separated batch sizes to probe.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for forward pass. CPU works but is slower for big models.")
    p.add_argument("--measure-backward", action="store_true",
                   help="Run forward + backward so peak (CUDA) reflects training memory.")
    p.add_argument("--input-shape", default="hbn", choices=["hbn", "native"],
                   help="hbn: (B,129,1600); native: each model's pretrained shape.")
    p.add_argument("--no-download", action="store_true",
                   help="Skip HF download — uses architecture only (random init). "
                        "Useful when you only need parameter counts.")
    p.add_argument("--out-dir", default="scripts/estimate_memory_output",
                   help="Where to write CSV + PNG.")
    return p.parse_args()


def select_models(arg: str) -> list[tuple[str, int, int, int, str]]:
    if arg == "all":
        return list(BACKBONES)
    wanted = {s.strip() for s in arg.split(",")}
    known = {b[0] for b in BACKBONES}
    missing = wanted - known
    if missing:
        sys.exit(f"Unknown model(s): {missing}. Known: {known}")
    return [b for b in BACKBONES if b[0] in wanted]


def load_backbone(factory_name: str, download: bool) -> nn.Module:
    """Build an OEB backbone at the HBN input shape. ``download=True`` also
    loads HuggingFace pretrained weights via ``PretrainedBackbone.load_pretrained``.
    """
    try:
        from open_eeg_bench.default_configs import backbones as oeb_backbones
    except ImportError as exc:
        raise ImportError(
            "open_eeg_bench is not installed. Run:\n"
            "    uv run --isolated --with open-eeg-bench --prerelease=allow "
            "scripts/estimate_memory.py …\n"
            "or `uv add open-eeg-bench --prerelease=allow` if your project deps "
            "are compatible."
        ) from exc

    if not hasattr(oeb_backbones, factory_name):
        available = sorted(
            n for n in dir(oeb_backbones)
            if not n.startswith("_")
            and callable(getattr(oeb_backbones, n, None))
            and n.islower()  # factory functions are lowercase; classes are TitleCase
        )
        raise AttributeError(
            f"OEB has no factory '{factory_name}' (likely not in your installed OEB version). "
            f"Available factories: {available}"
        )

    factory: Callable = getattr(oeb_backbones, factory_name)
    pb = factory()  # PretrainedBackbone (Pydantic spec, not an nn.Module)

    chs_info = _build_chs_info()
    model = pb.build(
        n_chans=HBN_N_CHANS,
        n_times=HBN_N_TIMES_FLAT,
        n_outputs=1,
        sfreq=HBN_SFREQ,
        chs_info=chs_info,
    )
    if download:
        try:
            pb.load_pretrained(model)
        except Exception as exc:
            print(f"  ! load_pretrained failed for {factory_name} ({exc}); using random init")
    return model


def report_model(name: str, model: nn.Module, args, native_shape: tuple[int, int]) -> list[dict]:
    """Print static memory and probe each batch size; return per-row dicts."""
    chans, times = (HBN_N_CHANS, HBN_N_TIMES_FLAT) if args.input_shape == "hbn" else native_shape

    # Materialize lazy modules (e.g. CBraMod uses LazyLinear) before counting params.
    model.to(args.device)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(make_input(1, chans, times, args.device))
    except Exception as exc:
        print(f"  ! dummy forward failed for {name} on shape (1, {chans}, {times}): {exc}")
        return []

    trainable, total = count_parameters(model)
    stat = static_memory(trainable)

    print(f"\n=== {name} ===")
    print(f"  params:       {total:>12,d} total ({trainable:,d} trainable)")
    print(f"  input shape:  (B, {chans}, {times})   [{'HBN' if args.input_shape == 'hbn' else 'native'}]")
    print(f"  weights:      {human_bytes(stat['weights'])}")
    print(f"  gradients:    {human_bytes(stat['gradients'])}")
    print(f"  AdamW state:  {human_bytes(stat['adam_state'])}  (m + v, fp32)")
    print(f"  static total: {human_bytes(stat['total_static'])}")

    rows = []
    model.to(args.device)

    print(f"\n  {'batch':>6} | {'input':>10} | {'act (est)':>12} | {'cuda peak':>12} | {'train total':>12}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for bs in args.batch_sizes:
        input_bytes = bs * chans * times * DTYPE_BYTES
        try:
            act_bytes, cuda_peak = try_forward(model, bs, chans, times, args.device, args.measure_backward)
            err = None
        except Exception as exc:
            act_bytes, cuda_peak, err = 0, None, str(exc).splitlines()[0][:80]

        train_total = stat["total_static"] + input_bytes + act_bytes
        rows.append({
            "model": name,
            "batch_size": bs,
            "params": trainable,
            "weights_bytes": stat["weights"],
            "grad_bytes": stat["gradients"],
            "adam_bytes": stat["adam_state"],
            "static_bytes": stat["total_static"],
            "input_bytes": input_bytes,
            "activation_bytes_est": act_bytes,
            "cuda_peak_bytes": cuda_peak or 0,
            "train_total_bytes": train_total,
            "error": err or "",
        })
        marker = f"  ✗ {err}" if err else ""
        cuda_str = human_bytes(cuda_peak) if cuda_peak else "    —    "
        print(f"  {bs:>6} | {human_bytes(input_bytes):>10} | {human_bytes(act_bytes):>12} | "
              f"{cuda_str:>12} | {human_bytes(train_total):>12}{marker}")

        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return rows


def write_outputs(rows: list[dict], out_dir: Path):
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "memory_estimates.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for name, group in by_model.items():
        group.sort(key=lambda r: r["batch_size"])
        xs = [r["batch_size"] for r in group]
        ys = [r["train_total_bytes"] / (1024**3) for r in group]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Estimated training memory (GiB)")
    ax.set_title("HBN pretraining memory by backbone\n(static fp32 AdamW + input + activation estimate)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    for ref in (8, 16, 24, 40, 80):  # common GPU VRAMs in GiB
        ax.axhline(ref, color="gray", ls="--", lw=0.6, alpha=0.5)
        ax.text(ax.get_xlim()[1], ref, f"{ref} GiB", va="center", ha="right", color="gray", fontsize=8)
    png_path = out_dir / "memory_estimates.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print(f"Wrote {png_path}")


def main():
    args = parse_args()
    args.batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    selected = select_models(args.models)

    print(f"HBN input:       n_chans={HBN_N_CHANS}, sfreq={HBN_SFREQ}Hz, "
          f"window={HBN_WINDOW_SECONDS}s, n_windows={HBN_N_WINDOWS}")
    print(f"  per-sample:    ({HBN_N_CHANS}, {HBN_N_TIMES_FLAT}) fp32 = "
          f"{human_bytes(HBN_N_CHANS * HBN_N_TIMES_FLAT * DTYPE_BYTES)}")
    print(f"  per-window:    ({HBN_N_CHANS}, {HBN_N_TIMES_PER_WINDOW}) fp32 = "
          f"{human_bytes(HBN_N_CHANS * HBN_N_TIMES_PER_WINDOW * DTYPE_BYTES)}")
    print(f"  features/sample: ({HBN_N_WINDOWS}, {HBN_FEATURES}) fp32 = "
          f"{human_bytes(HBN_N_WINDOWS * HBN_FEATURES * DTYPE_BYTES)}")
    print(f"\nDevice:          {args.device}{'  (backward measured)' if args.measure_backward else ''}")
    print(f"Input shape:     {args.input_shape}")
    print(f"Batch sizes:     {args.batch_sizes}\n")
    print(f"Static memory uses fp32 AdamW: 4N(weights) + 4N(grad) + 8N(m+v) = 16N bytes.")
    print(f"With bf16 mixed-precision + fp32 master, the typical factor is ~18N "
          f"(add 2N for bf16 copy) — multiply 'static total' by ~1.125 for that case.\n")

    all_rows = []
    for factory_name, n_c, n_t, _sf, note in selected:
        print(f"\nLoading backbone '{factory_name}' …  ({note})")
        try:
            model = load_backbone(factory_name, download=not args.no_download)
        except Exception as exc:
            print(f"  ! failed to instantiate {factory_name}: {exc}")
            continue
        try:
            rows = report_model(factory_name, model, args, native_shape=(n_c, n_t))
            all_rows.extend(rows)
        finally:
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    if all_rows:
        write_outputs(all_rows, Path(args.out_dir))
    else:
        print("\nNo successful runs — nothing to write.")


if __name__ == "__main__":
    main()
