# Position-leakage diagnostics

Tests whether learned encoders leak `position_in_movie` (time within
the clip) in a way the supervised probe can lift, and provides trivial
baselines so probe scores have a calibrated chance level.

## Why this study exists

EEG drifts within a recording (electrode impedance, attentional state,
body movement). If the encoder picks up that drift, a probe targeting
`position_in_movie` will look like it's doing real stimulus decoding
when it's actually just reading recording-time. The trivial baselines
quantify how much a model could score without seeing the stimulus at all.

## Entry points

- [`baseline.py`](baseline.py) -- "predict position from raw EEG band-power
  / shuffled features / aligned-across-subjects features" baselines.
- [`sweeps/run_delta.py`](sweeps/run_delta.py) -- launches the position
  leakage probe-eval sweep on Delta against a list of trained checkpoints.

## Run

```bash
PYTHONPATH=. uv run --group eeg python experiments/position_leakage/baseline.py
```

## Related

The `--shuffle_position_within_rec` flag on
`eb_jepa.evaluation.run_probe_eval` is the in-pipeline counterpart: it
shuffles position labels within each recording so a model that only
encodes within-recording temporal drift drops to ~0 probe corr.
