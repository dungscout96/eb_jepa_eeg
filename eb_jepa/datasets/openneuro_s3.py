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


def _ch_mismatch_strategy() -> str:
    """Pick the least-invasive ``on_ch_mismatch`` the installed mne-bids accepts.

    Version-dependent, and getting this wrong costs a whole release: mne-bids
    0.19 accepts ``"warn"`` (skip the channels.tsv metadata and carry on), but
    0.18 -- which is what Delta has -- validates against
    ``{'reorder','raise','rename'}`` and raises ValueError on "warn". Reading
    the 0.19 source locally and assuming it applied to the cluster is exactly
    how R7/DespicableMe failed a second time.

    Order of preference:
      "warn"    least invasive: ignores the mismatched tsv metadata entirely.
      "rename"  renames raw channels to the tsv names. Fires only on an actual
                mismatch, and ``drop_anomalous_recordings`` still removes the
                offending recording by channel count afterwards.
    """
    import inspect

    import mne_bids.read as _read

    try:
        src = inspect.getsource(_read._handle_channel_mismatch)
    except Exception:  # noqa: BLE001
        return "rename"
    return "warn" if '"warn"' in src or "'warn'" in src else "rename"


def repair_channel_types(ds_root: Path, default_type: str = "EEG") -> int:
    """Fill an empty ``type`` column in mirrored ``channels.tsv`` files.

    ds005515 (R10) ships every ``channels.tsv`` with the ``type`` column blank,
    while R7/R8/R9 all say ``EEG``. mne-bids overrides channel types from that
    file, so blank types mean nothing is typed as EEG and the first
    ``raw.filter()`` dies with "picks (data_or_ica) yielded no channels" -- deep
    inside pass 1, long after the 22 GB download.

    Safe because the ``.set`` itself already types all 129 channels as eeg and
    the names match the TSV exactly; this restores what the recording says about
    itself. Only the local mirror is modified.

    Returns the number of files repaired.
    """
    import pandas as pd

    n = 0
    for tsv in ds_root.rglob("*_channels.tsv"):
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception:
            continue
        if "type" not in df.columns or df["type"].notna().any():
            continue
        df["type"] = default_type
        df.to_csv(tsv, sep="\t", index=False, na_rep="n/a")
        n += 1
    if n:
        logger.warning(
            "Repaired %d channels.tsv file(s) under %s with a blank `type` "
            "column (filled with %r from the recording's own typing).",
            n, ds_root.name, default_type)
    return n


def drop_anomalous_recordings(dataset) -> list[str]:
    """Remove recordings whose channel count differs from the cohort mode.

    ds005511 (R7) contains sub-NDARBA381JGH's DespicableMe file with 6 unnamed
    channels ("EEG 000"...) where 129 are expected -- a corrupt recording that
    otherwise aborts the whole release-task. Dropping by *modal* channel count
    rather than a hardcoded 129 keeps this honest if a future release legitimately
    uses a different montage.

    Returns the subject labels dropped.
    """
    from collections import Counter

    counts = [len(d.raw.ch_names) for d in dataset.datasets]
    if not counts:
        return []
    modal, _ = Counter(counts).most_common(1)[0]
    keep, dropped = [], []
    for d, c in zip(dataset.datasets, counts):
        if c == modal:
            keep.append(d)
        else:
            desc = d.description
            dropped.append(f"{desc.get('subject', '?')}({c}ch)")
    if dropped:
        logger.warning(
            "Dropping %d recording(s) whose channel count != %d (the cohort "
            "mode): %s", len(dropped), modal, ", ".join(dropped))
        dataset.datasets = keep
    return dropped


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
    repair_channel_types(ds_root)
    # Anything but the default "raise": a single corrupt recording (R7
    # sub-NDARBA381JGH, 6 unnamed channels) otherwise aborts the entire
    # release-task. The accepted values differ by mne-bids version, hence the
    # probe. Mismatched files fall through to drop_anomalous_recordings below
    # rather than being silently kept.
    strategy = _ch_mismatch_strategy()
    logger.info("on_ch_mismatch=%r (mne-bids-version dependent)", strategy)
    ds = BIDSDataset(root=ds_root, tasks=task, datatypes="eeg", preload=False,
                     on_ch_mismatch=strategy)
    drop_anomalous_recordings(ds)
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
