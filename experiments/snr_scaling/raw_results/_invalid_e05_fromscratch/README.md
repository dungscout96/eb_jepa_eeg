# Invalid — the first e05_addval sweep, trained from scratch

These 28 artifacts are the E0.6 arm as first submitted (2026-08-14). They are
**not comparable to e04** and must not be read into any table.

The submitter reconstructed e04's recipe from its on-disk `config.yaml`, which
records `meta.encoder_init_from: null`. The actual e04 runs override that on
the CLI — every cell warm-starts its encoder from
`/work/hdd/bbnv/kkokate/eb_jepa/reve_base_eet_init.pth.tar` (confirmed in 5/5
sampled cells' `wandb/latest-run/files/wandb-metadata.json`), and also uses
`--meta.seed=2026` rather than the file's 2025. A byte-for-byte diff of the two
arms' config.yaml therefore came back clean while the arms differed in the
single most important way.

What that looked like in the numbers, and why it was not mistaken for a result:

- ~37 % below e04 at *matched* S=1400 (TP probe mean r 0.1853 vs 0.2927), i.e.
  -12.6 between-draw sd, on every one of five independent readouts.
- The S curve went completely FLAT: 0.1853 (S=1400), 0.1811 (S=1863), 0.1859
  (S=2156). A from-scratch depth-22 encoder at 4400 steps is under-trained
  enough that subject count stops mattering.
- In-loop `clip_scene_auc` plateaued at ~0.72 against e04's ~0.84 — while R5
  was *in* these cells' training set, so that metric was if anything inflated.

Kept rather than deleted so the failure is reproducible and so nobody re-runs
the same sweep to rediscover it. The corrected arm overwrites the checkpoints
in place on Delta and rewrites `raw_results/*_e05_*.json` one directory up.
