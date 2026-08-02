"""Fetch OpenNeuro BIDS data straight from S3, bypassing the eegdash API.

``eegdash`` resolves recordings through ``data.eegdash.org``. When that service
is unreachable -- as during the 2026-08-02 outage, which failed identically from
Delta and from a laptop, so it was the service and not a cluster firewall --
nothing can be downloaded, even though OpenNeuro's S3 bucket is fine.

This module goes to S3 directly. It mirrors only the files one (dataset, task)
actually needs into a local BIDS tree, then hands that tree to
``braindecode.datasets.BIDSDataset``. Reusing BIDSDataset rather than building
``BaseDataset`` objects by hand means mne-bids still supplies ``bidspath``, so
``preprocess_hbn.py``'s fast sidecar-duration path keeps working instead of
falling back to opening every raw.

Verified before writing this: reading an HBN ``.set`` directly preserves the
``video_start`` / ``video_stop`` annotations that
``get_movie_recording_duration`` depends on (checked on
ds005512/sub-NDARAA306NT2: 500 Hz, 129 channels, movie duration 203.28 s).

Downloads are resumable -- a file whose size already matches the S3 object is
skipped -- so an interrupted job re-runs cheaply.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

S3_ROOT = "https://s3.amazonaws.com/openneuro.org"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Root-level BIDS files worth mirroring. Small, and mne-bids/BIDSDataset expect
# at least dataset_description.json to consider the tree valid.
_ROOT_FILES = (
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "CHANGES",
)

# Per-recording sidecars. The .set carries the data (HBN embeds it rather than
# using a separate .fdt), but .fdt is fetched when present.
_RECORDING_SUFFIXES = ("_eeg.set", "_eeg.fdt", "_eeg.json",
                       "_channels.tsv", "_events.tsv", "_events.json")


def _get(url: str, timeout: int = 120, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as exc:                      # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def list_objects(prefix: str) -> dict[str, int]:
    """Return ``{key: size}`` for every S3 object under *prefix*.

    Paginated: S3 caps a listing at 1000 keys and signals more with
    ``IsTruncated``. Without following the continuation token a release with
    many subjects would silently list only its first few.
    """
    out: dict[str, int] = {}
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        xml = _get(f"{S3_ROOT}?{urllib.parse.urlencode(q)}")
        root = ET.fromstring(xml)
        for c in root.findall(f"{_S3_NS}Contents"):
            key = c.findtext(f"{_S3_NS}Key")
            size = int(c.findtext(f"{_S3_NS}Size") or 0)
            if key:
                out[key] = size
        if (root.findtext(f"{_S3_NS}IsTruncated") or "false").lower() != "true":
            break
        token = root.findtext(f"{_S3_NS}NextContinuationToken")
        if not token:
            break
    return out


def subjects_with_task(dataset_id: str, task: str) -> list[str]:
    """Subject labels having a recording for *task*, from the S3 listing.

    Matches the full ``_task-<task>_eeg.set`` suffix rather than a substring, so
    a task name that is a prefix of another cannot pull in the wrong files.
    """
    keys = list_objects(f"{dataset_id}/sub-")
    pat = re.compile(rf"^{re.escape(dataset_id)}/(sub-[^/]+)/eeg/"
                     rf"\1(_ses-[^_]+)?_task-{re.escape(task)}"
                     rf"(_run-\d+)?_eeg\.set$")
    subs = {m.group(1) for k in keys if (m := pat.match(k))}
    return sorted(subs)


def _download(key: str, dest: Path, expected_size: int | None) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        return "skip"
    tmp = dest.with_suffix(dest.suffix + ".part")
    data = _get(f"{S3_ROOT}/{urllib.parse.quote(key)}")
    tmp.write_bytes(data)
    os.replace(tmp, dest)       # atomic: a killed job never leaves a short file
    return "get"


def fetch_bids_task(dataset_id: str, task: str, root: Path,
                    *, max_workers: int = 8,
                    subjects: list[str] | None = None) -> Path:
    """Mirror the files for one (dataset, task) into a local BIDS tree at *root*.

    Returns the dataset root (``root / dataset_id``) for BIDSDataset.
    """
    ds_root = Path(root) / dataset_id
    ds_root.mkdir(parents=True, exist_ok=True)

    all_keys = list_objects(f"{dataset_id}/")
    for name in _ROOT_FILES:
        key = f"{dataset_id}/{name}"
        if key in all_keys:
            _download(key, ds_root / name, all_keys[key])

    subs = subjects if subjects is not None else subjects_with_task(dataset_id, task)
    if not subs:
        raise RuntimeError(
            f"No subjects found for {dataset_id} task={task}. Check the task "
            "spelling against the S3 listing.")
    logger.info("%s/%s: %d subjects to mirror", dataset_id, task, len(subs))

    # Trailing underscore matters: "_task-ThePresent" alone also matches
    # "_task-ThePresentExtra". Every suffix in _RECORDING_SUFFIXES starts with
    # "_", so the entity is always followed by one.
    sub_set = set(subs)
    tag = f"_task-{task}_"
    wanted = [
        k for k in all_keys
        if tag in k
        and k.endswith(_RECORDING_SUFFIXES)
        and k.split("/")[1] in sub_set
    ]
    total_bytes = sum(all_keys[k] for k in wanted)
    logger.info("%s/%s: %d files, %.1f GB", dataset_id, task, len(wanted),
                total_bytes / 1e9)

    done = {"get": 0, "skip": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_download, k, ds_root / Path(k).relative_to(dataset_id),
                      all_keys[k]): k
            for k in wanted
        }
        for i, fut in enumerate(as_completed(futs), 1):
            done[fut.result()] += 1
            if i % 200 == 0 or i == len(futs):
                logger.info("  %d/%d files (%d fetched, %d already present)",
                            i, len(futs), done["get"], done["skip"])
    return ds_root


def load_from_s3(release: str, task: str, cache_dir: Path,
                 *, max_workers: int = 8):
    """eegdash-free replacement for ``load_or_download``.

    Mirrors the (release, task) subtree from S3, then returns a
    ``BIDSDataset`` over it -- the same ``BaseConcatDataset`` interface
    ``preprocess_hbn.py`` already consumes.
    """
    from braindecode.datasets import BIDSDataset

    from eb_jepa.datasets.hbn import _release_to_dataset_id

    dataset_id = _release_to_dataset_id(release)
    ds_root = fetch_bids_task(dataset_id, task, Path(cache_dir),
                              max_workers=max_workers)
    ds = BIDSDataset(root=ds_root, tasks=task, datatypes="eeg", preload=False)
    normalise_descriptions(ds, task)
    logger.info("Loaded %d recordings from %s (task=%s)",
                len(ds.datasets), ds_root, task)
    return ds


def normalise_descriptions(dataset, task: str) -> None:
    """Reduce each recording's description to ``{subject, task}``, in place.

    Two reasons, both load-bearing:

    1. ``BIDSDataset`` puts a ``PosixPath`` under ``path``, and braindecode
       serialises the description to ``description.json`` when saving. That
       raises ``TypeError: Object of type PosixPath is not JSON serializable``
       -- at *save* time, i.e. after the expensive resample/filter pass has
       already run.
    2. The eegdash path produced exactly ``{"subject": ..., "task": ...}``
       (verified against R5's on-disk ``description.json``). Matching it keeps
       releases mirrored via S3 indistinguishable from R1-R6 downstream.
    """
    import pandas as pd

    for d in dataset.datasets:
        desc = d.description
        subject = desc.get("subject") if hasattr(desc, "get") else None
        # NOT set_description(): it MERGES (pops colliding keys, then concats),
        # so keys we omit -- including the offending `path` -- would survive and
        # the JSON save would still fail. Replacing the Series is the only way
        # to actually drop them.
        d._description = pd.Series(
            {"subject": subject, "task": desc.get("task", task)}
        )
