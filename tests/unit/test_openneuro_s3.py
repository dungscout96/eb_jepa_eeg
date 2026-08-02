"""Tests for the S3 fallback that bypasses the eegdash API.

The properties worth pinning are the ones whose failure is SILENT: a paginated
listing that stops at 1000 keys would quietly drop most of a large release, and
a loose task match would mirror the wrong recordings. Both produce a dataset
that looks fine and is wrong.

Network is stubbed throughout -- these must not hit S3.
"""

import urllib.parse
from pathlib import Path

import pytest

from eb_jepa.datasets import openneuro_s3 as s3

NS = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'


def _xml(keys_sizes, truncated=False, token=None):
    body = "".join(
        f"<Contents><Key>{k}</Key><Size>{v}</Size></Contents>"
        for k, v in keys_sizes
    )
    tok = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (f'<?xml version="1.0"?><ListBucketResult {NS}>{body}'
            f"<IsTruncated>{str(truncated).lower()}</IsTruncated>{tok}"
            "</ListBucketResult>").encode()


# ------------------------------------------------------------------ listing


def test_list_objects_follows_pagination(monkeypatch):
    """S3 caps a listing at 1000 keys. Not following the continuation token
    would silently mirror only part of a large release."""
    pages = [
        _xml([("ds1/a", 1), ("ds1/b", 2)], truncated=True, token="TOK1"),
        _xml([("ds1/c", 3)], truncated=False),
    ]
    seen = []

    def fake_get(url, **kw):
        seen.append(url)
        return pages[len(seen) - 1]

    monkeypatch.setattr(s3, "_get", fake_get)
    out = s3.list_objects("ds1/")
    assert out == {"ds1/a": 1, "ds1/b": 2, "ds1/c": 3}
    assert len(seen) == 2
    assert "continuation-token=TOK1" in urllib.parse.unquote(seen[1])


def test_list_objects_stops_when_truncated_but_token_missing(monkeypatch):
    """A malformed response must terminate, not spin forever."""
    monkeypatch.setattr(s3, "_get",
                        lambda url, **kw: _xml([("ds1/a", 1)], truncated=True))
    assert s3.list_objects("ds1/") == {"ds1/a": 1}


# ------------------------------------------------------------------ matching


def test_subjects_with_task_matches_full_task_token(monkeypatch):
    """A substring match would pull ThePresent files into a hypothetical
    'Present' task, and vice versa. Match the whole _task-<name>_eeg.set."""
    keys = [
        ("ds1/sub-A/eeg/sub-A_task-ThePresent_eeg.set", 10),
        ("ds1/sub-B/eeg/sub-B_task-ThePresent_eeg.set", 10),
        ("ds1/sub-C/eeg/sub-C_task-DespicableMe_eeg.set", 10),
        ("ds1/sub-D/eeg/sub-D_task-ThePresentExtra_eeg.set", 10),
        ("ds1/sub-E/eeg/sub-E_task-ThePresent_channels.tsv", 10),  # not a .set
    ]
    monkeypatch.setattr(s3, "_get", lambda url, **kw: _xml(keys))
    assert s3.subjects_with_task("ds1", "ThePresent") == ["sub-A", "sub-B"]


def test_subjects_with_task_accepts_session_and_run_entities(monkeypatch):
    keys = [
        ("ds1/sub-A/eeg/sub-A_ses-1_task-TP_eeg.set", 10),
        ("ds1/sub-B/eeg/sub-B_task-TP_run-2_eeg.set", 10),
    ]
    monkeypatch.setattr(s3, "_get", lambda url, **kw: _xml(keys))
    assert s3.subjects_with_task("ds1", "TP") == ["sub-A", "sub-B"]


# ------------------------------------------------------------------ download


def test_download_skips_when_size_already_matches(tmp_path, monkeypatch):
    """Resume: an interrupted mirror must not re-fetch gigabytes."""
    dest = tmp_path / "f.set"
    dest.write_bytes(b"xxxxx")
    monkeypatch.setattr(s3, "_get",
                        lambda *a, **k: pytest.fail("should not download"))
    assert s3._download("ds1/f.set", dest, 5) == "skip"


def test_download_refetches_when_size_differs(tmp_path, monkeypatch):
    dest = tmp_path / "f.set"
    dest.write_bytes(b"xx")                      # truncated from a killed job
    monkeypatch.setattr(s3, "_get", lambda *a, **k: b"xxxxx")
    assert s3._download("ds1/f.set", dest, 5) == "get"
    assert dest.read_bytes() == b"xxxxx"


def test_download_is_atomic(tmp_path, monkeypatch):
    """A killed job must never leave a short file that a later resume would
    accept -- write to .part, then rename."""
    dest = tmp_path / "sub" / "f.set"
    observed = {}

    def fake_get(url, **kw):
        observed["parts_before_rename"] = list(dest.parent.glob("*.part")) \
            if dest.parent.exists() else []
        return b"data"

    monkeypatch.setattr(s3, "_get", fake_get)
    s3._download("ds1/sub/f.set", dest, None)
    assert dest.read_bytes() == b"data"
    assert not list(dest.parent.glob("*.part"))


def test_get_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise OSError("no route to host")

    monkeypatch.setattr(s3.urllib.request, "urlopen", boom)
    monkeypatch.setattr(s3.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="GET failed after 4 tries"):
        s3._get("https://example/x", retries=4)
    assert calls["n"] == 4


# ------------------------------------------------------------------ fetch


def test_fetch_bids_task_raises_on_empty_subject_set(tmp_path, monkeypatch):
    """Silently producing an empty dataset would look like a release with no
    recordings for the task, rather than a typo."""
    monkeypatch.setattr(s3, "list_objects", lambda p: {})
    monkeypatch.setattr(s3, "subjects_with_task", lambda d, t: [])
    with pytest.raises(RuntimeError, match="No subjects found"):
        s3.fetch_bids_task("ds1", "Nope", tmp_path)


def test_fetch_bids_task_mirrors_only_the_requested_task(tmp_path, monkeypatch):
    keys = {
        "ds1/dataset_description.json": 5,
        "ds1/participants.tsv": 5,
        "ds1/sub-A/eeg/sub-A_task-TP_eeg.set": 10,
        "ds1/sub-A/eeg/sub-A_task-TP_events.tsv": 3,
        "ds1/sub-A/eeg/sub-A_task-OTHER_eeg.set": 10,   # must NOT be fetched
        "ds1/sub-Z/eeg/sub-Z_task-TP_eeg.set": 10,      # subject not selected
    }
    monkeypatch.setattr(s3, "list_objects", lambda p: keys)
    monkeypatch.setattr(s3, "subjects_with_task", lambda d, t: ["sub-A"])
    monkeypatch.setattr(s3, "_get", lambda url, **kw: b"x" * 3)

    root = s3.fetch_bids_task("ds1", "TP", tmp_path)
    got = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert "sub-A/eeg/sub-A_task-TP_eeg.set" in got
    assert "sub-A/eeg/sub-A_task-TP_events.tsv" in got
    assert "dataset_description.json" in got
    assert not any("OTHER" in g for g in got)
    assert not any("sub-Z" in g for g in got)


def test_fetch_bids_task_does_not_mirror_a_task_with_the_name_as_prefix(
        tmp_path, monkeypatch):
    """A bare "_task-ThePresent" substring also matches "_task-ThePresentExtra".
    Wasteful rather than wrong (BIDSDataset filters at load), but on a 19 GB
    release the waste is the point."""
    keys = {
        "ds1/sub-A/eeg/sub-A_task-TP_eeg.set": 10,
        "ds1/sub-A/eeg/sub-A_task-TPExtra_eeg.set": 10,
    }
    monkeypatch.setattr(s3, "list_objects", lambda p: keys)
    monkeypatch.setattr(s3, "subjects_with_task", lambda d, t: ["sub-A"])
    monkeypatch.setattr(s3, "_get", lambda url, **kw: b"xxx")

    root = s3.fetch_bids_task("ds1", "TP", tmp_path)
    got = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert "sub-A/eeg/sub-A_task-TP_eeg.set" in got
    assert not any("TPExtra" in g for g in got)


# ------------------------------------------------------- description schema


class _FakeDS:
    def __init__(self, desc):
        import pandas as pd
        self._description = pd.Series(desc)

    @property
    def description(self):
        return self._description


class _FakeConcat:
    def __init__(self, datasets):
        self.datasets = datasets


def test_normalise_descriptions_drops_the_unserialisable_path():
    """BIDSDataset puts a PosixPath under `path`; braindecode serialises the
    description to JSON when saving, which raises TypeError -- and it raises
    AFTER the expensive resample/filter pass has run. Must be dropped."""
    import json
    ds = _FakeConcat([_FakeDS({
        "path": Path("/a/b_eeg.set"), "subject": "NDAR1", "session": None,
        "task": "ThePresent", "run": None, "extension": ".set",
    })])
    s3.normalise_descriptions(ds, "ThePresent")
    d = dict(ds.datasets[0].description)
    assert d == {"subject": "NDAR1", "task": "ThePresent"}
    json.dumps(d)          # the thing that used to blow up


def test_normalise_descriptions_matches_the_eegdash_schema_exactly():
    """R5's on-disk description.json is exactly {"subject","task"}. Releases
    mirrored via S3 must be indistinguishable from R1-R6 downstream."""
    ds = _FakeConcat([_FakeDS({"path": Path("/x.set"), "subject": "S", "task": "TP"})])
    s3.normalise_descriptions(ds, "TP")
    assert sorted(dict(ds.datasets[0].description)) == ["subject", "task"]


def test_normalise_descriptions_falls_back_to_the_requested_task():
    ds = _FakeConcat([_FakeDS({"subject": "S"})])
    s3.normalise_descriptions(ds, "DespicableMe")
    assert dict(ds.datasets[0].description)["task"] == "DespicableMe"
