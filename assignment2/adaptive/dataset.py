"""Prepare per-cutoff snapshots from official Parquet or ZIP files."""

import hashlib
import json
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from relational.data import PRIMARY, Store, digest, read_snapshot, utc

from .features import DAY, SCHEMA_VERSION, snapshot
from .gnn import graph_snapshot


@contextmanager
def new_directory(path):
    """Publish the complete run only after all writes succeed."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        work = Path(tmp) / "result"
        work.mkdir()
        yield work
        work.rename(path)


def read_ids(path):
    if Path(path).suffix == ".csv":
        ids = pd.read_csv(path, dtype={"UserID": "int64"}).UserID.to_numpy()
    else:
        ids = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if not len(ids) or len(np.unique(ids)) != len(ids) or (ids < 0).any():
        raise ValueError("Expected a nonempty list of unique user IDs")
    return ids


def load_sources(paths, cohort, until):
    """Resolve keys within each archive before combining its rows."""
    rows, audits = [], []
    for number, path in enumerate(map(Path, paths)):
        with tempfile.TemporaryDirectory() as tmp:
            folder = path
            if path.is_file():
                folder = Path(tmp)
                with zipfile.ZipFile(path) as archive:
                    for table in PRIMARY:
                        name = table + ".parquet"
                        matches = [
                            x for x in archive.namelist() if Path(x).name == name
                        ]
                        if len(matches) != 1:
                            raise ValueError(f"Expected one {name} in {path}")
                        with archive.open(matches[0]) as src:
                            with (folder / name).open("wb") as dst:
                                shutil.copyfileobj(src, dst)
            part, audit = read_snapshot(folder, number, before=until)
            rows.append(part)
            audits.extend(audit)
    return Store(pd.concat(rows, ignore_index=True), cohort), audits


def read_coverage(path):
    if path is None:
        return None
    frame = pd.read_csv(path)
    for key in ("hour_utc", "available_at"):
        frame[key] = pd.to_datetime(frame[key], utc=True).dt.tz_localize(None)
    if (
        not frame.observed.isin([0, 1]).all()
        or frame[["hour_utc", "available_at"]].isna().any().any()
        or frame.duplicated(["hour_utc", "available_at"]).any()
        or not frame.hour_utc.eq(frame.hour_utc.dt.floor("h")).all()
    ):
        raise ValueError("Invalid hourly observation manifest")
    return frame


def calendar(plan):
    train = plan["train"]
    if train["stride_days"] < 1:
        raise ValueError("stride_days must be positive")
    dates = dict(
        train=list(
            pd.date_range(
                utc(train["first"]),
                utc(train["last"]),
                freq=f"{train['stride_days']}D",
            )
        )
    )
    for split in ("stop", "calibration", "evaluation", "test"):
        if split != "test" or split in plan:
            dates[split] = [utc(x) for x in plan[split]]
    days, cap = plan["history_days"], plan["neighbors"]
    if days < 1 or cap < 1:
        raise ValueError("history_days and neighbors must be positive")
    previous_end = None
    for split, values in dates.items():
        if not values or values != sorted(set(values)):
            raise ValueError(f"Expected ordered distinct cutoffs in {split}")
        if any(t != t.floor("h") for t in values):
            raise ValueError("Cutoffs must be aligned to an hour")
        if previous_end is not None and values[0] < previous_end:
            raise ValueError(f"Overlapping label windows before {split}")
        previous_end = values[-1] + 7 * DAY
    if dates["train"][0] - max(14, days) * DAY < utc(
        plan["observation_start"]
    ) or previous_end > utc(plan["observation_end"]):
        raise ValueError("History or label window outside observation bounds")
    return dates


def prepare(args):
    plan = json.loads(args.plan.read_text())
    dates = calendar(plan)
    cohort = read_ids(args.cohort)
    ids = read_ids(args.users) if args.users else cohort
    if args.sample_users is not None:
        if not 0 < args.sample_users <= len(ids):
            raise ValueError("sample-users outside cohort size")
        rng = np.random.default_rng(args.seed)
        ids = np.sort(rng.choice(ids, size=args.sample_users, replace=False))
    if not np.isin(ids, cohort).all():
        raise ValueError("Labeled users must belong to the cohort")
    coverage = read_coverage(args.coverage)
    with new_directory(args.data) as folder:
        store, audits = load_sources(args.db, cohort, plan["observation_end"])
        clock_max = store.rows.known.max()
        if pd.isna(clock_max) or clock_max < max(v[-1] for v in dates.values()):
            raise ValueError("Input archives end before the last label week")
        pd.DataFrame({"UserID": ids}).to_csv(folder / "users.csv", index=False)
        meta = dict(
            schema=SCHEMA_VERSION,
            gnn_schema="relative-time-v2",
            plan=plan,
            label_days=7,
            ids_sha256=digest(folder / "users.csv"),
            cohort_sha256=digest(args.cohort),
            n_users=len(ids),
            coverage_sha256=digest(args.coverage) if args.coverage else None,
            sources=[[a["source"], a["table"], a["sha256"]] for a in audits],
        )
        meta["data_id"] = hashlib.sha256(
            json.dumps(meta, sort_keys=True).encode()
        ).hexdigest()
        files = {}
        for split, values in dates.items():
            files[split] = []
            for t in values:
                name = f"{split}_{t:%Y%m%dT%H%M}.npz"
                print(f"Preparing {split} {t} UTC", flush=True)
                sample = snapshot(
                    store,
                    t,
                    ids,
                    plan["history_days"],
                    plan["neighbors"],
                    coverage,
                )
                graph = graph_snapshot(store, t, ids, plan["history_days"])
                for key in ("ids", "y", "p7", "p14"):
                    if not np.array_equal(sample[key], graph[key]):
                        raise ValueError(f"Adaptive/GNN mismatch: {key} at {t}")
                sample.update(
                    {
                        k: v
                        for k, v in graph.items()
                        if k not in ("ids", "y", "p7", "p14", "cutoff")
                    }
                )
                np.savez_compressed(folder / name, **sample)
                files[split].append(name)
        meta["files"] = files
        (folder / "manifest.json").write_text(json.dumps(meta, indent=2))
        (folder / "source_audit.json").write_text(json.dumps(audits, indent=2))
