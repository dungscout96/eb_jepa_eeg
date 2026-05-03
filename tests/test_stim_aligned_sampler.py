"""StimAlignedBatchSampler step-level iteration tests."""
import pytest


class _MockDataset:
    """Mock JEPAMovieDataset with the surface needed by StimAlignedBatchSampler."""
    def __init__(self, flat_clips, bucket_to_flat):
        self._flat_clips = flat_clips
        self._bucket_to_flat = bucket_to_flat
        self._stim_flat_index = True


def _build_mock(n_buckets=39, n_recs_per_bucket=50, clips_per_rec_per_bucket=20):
    """Build a synthetic dataset matching our actual smoke run scale."""
    flat_clips = []
    bucket_to_flat = {}
    for b in range(n_buckets):
        bucket_to_flat[(0, b)] = []
        for rec in range(n_recs_per_bucket):
            for c in range(clips_per_rec_per_bucket):
                idx = len(flat_clips)
                flat_clips.append((rec, c))
                bucket_to_flat[(0, b)].append(idx)
    return _MockDataset(flat_clips, bucket_to_flat)


def test_step_level_default_density_matches_random_sampler():
    """Default steps_per_epoch should equal n_clips // batch_size."""
    from eb_jepa.datasets.hbn import StimAlignedBatchSampler
    ds = _build_mock(n_buckets=39, n_recs_per_bucket=50, clips_per_rec_per_bucket=20)
    sampler = StimAlignedBatchSampler(ds, batch_size=64, K=2)
    expected = len(ds._flat_clips) // 64
    assert sampler.steps_per_epoch == expected
    assert len(sampler) == expected


def test_step_level_explicit_steps_per_epoch():
    from eb_jepa.datasets.hbn import StimAlignedBatchSampler
    ds = _build_mock()
    sampler = StimAlignedBatchSampler(ds, batch_size=64, K=2, steps_per_epoch=200)
    assert sampler.steps_per_epoch == 200
    assert len(sampler) == 200


def test_step_level_yields_correct_batch_count_and_size():
    from eb_jepa.datasets.hbn import StimAlignedBatchSampler
    ds = _build_mock()
    sampler = StimAlignedBatchSampler(ds, batch_size=64, K=2, steps_per_epoch=10)
    batches = list(iter(sampler))
    assert len(batches) == 10
    for b in batches:
        assert len(b) == 64


def test_step_level_pairs_share_bucket_and_distinct_recordings():
    """Within each batch, every consecutive pair must come from the same bucket
    AND from distinct recordings."""
    from eb_jepa.datasets.hbn import StimAlignedBatchSampler
    ds = _build_mock()
    sampler = StimAlignedBatchSampler(ds, batch_size=64, K=2, steps_per_epoch=20)
    for batch in sampler:
        # Pairs are (batch[2i], batch[2i+1]) per the K=2 yield order.
        for i in range(0, len(batch), 2):
            fi_a, fi_b = batch[i], batch[i + 1]
            rec_a, _ = ds._flat_clips[fi_a]
            rec_b, _ = ds._flat_clips[fi_b]
            assert rec_a != rec_b, f"pair {(fi_a, fi_b)} shares recording {rec_a}"
            # Find which bucket each clip belongs to
            bucket_a = next(k for k, v in ds._bucket_to_flat.items() if fi_a in v)
            bucket_b = next(k for k, v in ds._bucket_to_flat.items() if fi_b in v)
            assert bucket_a == bucket_b, f"pair {(fi_a, fi_b)} crosses buckets"


def test_step_level_set_epoch_changes_sampling():
    """set_epoch should produce a different batch sequence."""
    from eb_jepa.datasets.hbn import StimAlignedBatchSampler
    ds = _build_mock()
    sampler = StimAlignedBatchSampler(ds, batch_size=64, K=2, steps_per_epoch=5, seed=0)
    sampler.set_epoch(0)
    b0 = list(iter(sampler))
    sampler.set_epoch(1)
    b1 = list(iter(sampler))
    assert b0 != b1


def test_step_level_filters_buckets_with_too_few_recordings():
    """Buckets with fewer than K distinct recs must be excluded."""
    from eb_jepa.datasets.hbn import StimAlignedBatchSampler
    flat_clips = []
    bucket_to_flat = {}
    bucket_to_flat[(0, 0)] = []
    for rec in range(10):
        flat_clips.append((rec, 0))
        bucket_to_flat[(0, 0)].append(len(flat_clips) - 1)
    bucket_to_flat[(0, 1)] = []
    flat_clips.append((100, 0))
    bucket_to_flat[(0, 1)].append(len(flat_clips) - 1)
    ds = _MockDataset(flat_clips, bucket_to_flat)
    sampler = StimAlignedBatchSampler(ds, batch_size=4, K=2, steps_per_epoch=3)
    assert len(sampler.buckets) == 1
    assert sampler.buckets[0] == (0, 0)


def test_step_level_K4_batch_construction():
    """K=4 path: 16 buckets per batch, 4 distinct recs each."""
    from eb_jepa.datasets.hbn import StimAlignedBatchSampler
    ds = _build_mock(n_buckets=39, n_recs_per_bucket=50)
    sampler = StimAlignedBatchSampler(ds, batch_size=64, K=4, steps_per_epoch=5)
    for batch in sampler:
        assert len(batch) == 64
        for i in range(0, 64, 4):
            recs_in_group = {ds._flat_clips[batch[i + j]][0] for j in range(4)}
            assert len(recs_in_group) == 4
