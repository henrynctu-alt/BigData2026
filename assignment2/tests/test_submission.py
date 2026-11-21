"""End-to-end official preprocessing, seven models and submission parity."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from adaptive.dataset import read_ids
from adaptive.experiment import ALL_MODELS
from adaptive.gnn import HeteroSAGE, NeighborSampler, graph_snapshot
from official.scripts import preprocess
from official.scripts.evaluate import read_predictions, scores
from relational.data import Store
from run_adaptive import main


def call(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_adaptive.py", *map(str, args)])
    main()


def official_fixture(tmp_path):
    cohort = [preprocess.code_int("user", u) for u in (101, 202, 303)]
    lines = []
    n = 0
    for day in range(62):
        for actor in (101, 202, 303):
            if (day + actor) % 3 == 0:
                continue
            n += 1
            time = pd.Timestamp("2025-12-01", tz="UTC") + pd.Timedelta(
                days=day, hours=8
            )
            lines.append(
                json.dumps(
                    dict(
                        id=str(n),
                        type="PushEvent",
                        created_at=time.isoformat(),
                        actor=dict(id=actor, login=f"user{actor}"),
                        repo=dict(id=42, name="owner/repo"),
                        payload=dict(
                            push_id=n,
                            ref="refs/heads/main",
                            head="a" * 40,
                            before="b" * 40,
                            size=1,
                            distinct_size=1,
                            commits=[],
                        ),
                    )
                )
            )
    frames = preprocess.build_tables([lines], set(cohort))
    preprocess.write_output(tmp_path / "official_data", preprocess.anonymize(frames))
    cp = tmp_path / "cohort.txt"
    cp.write_text("\n".join(map(str, cohort)) + "\n")
    plan = json.loads(Path("configs/pilot.json").read_text())
    plan["test"] = ["2026-01-25T00:00:00+08:00"]
    plan["evaluation"] = ["2026-01-18T00:00:00+08:00"]
    plan["calibration"] = ["2026-01-11T00:00:00+08:00"]
    plan["stop"] = ["2026-01-04T00:00:00+08:00"]
    plan["train"]["last"] = plan["train"]["first"]
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(plan))
    return tmp_path / "official_data/db", cp, pp


def test_unified_official_to_submission(monkeypatch, tmp_path):
    db, cohort, plan = official_fixture(tmp_path)
    data, models, out = tmp_path / "cache", tmp_path / "models", tmp_path / "test"
    call(
        monkeypatch,
        "prepare",
        "--db",
        db,
        "--cohort",
        cohort,
        "--plan",
        plan,
        "--data",
        data,
    )
    call(
        monkeypatch,
        "train",
        "--data",
        data,
        "--out",
        models,
        "--epochs",
        1,
        "--hidden",
        8,
        "--gnn-hidden",
        8,
        "--gnn-fanout",
        2,
        "--batch-size",
        3,
        "--threads",
        1,
    )
    assert sorted(p.parent.name for p in models.glob("*/model.pt")) == sorted(
        ALL_MODELS
    )
    call(
        monkeypatch,
        "evaluate",
        "--data",
        data,
        "--models",
        models,
        "--split",
        "test",
        "--out",
        out,
        "--threads",
        1,
    )
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 7
    for name in ALL_MODELS:
        pred = tmp_path / f"{name}.csv"
        call(
            monkeypatch,
            "predict",
            "--checkpoint",
            models / name / "model.pt",
            "--db",
            db,
            "--cohort",
            cohort,
            "--users",
            data / "users.csv",
            "--cutoff",
            "2026-01-25T00:00:00+08:00",
            "--out",
            pred,
            "--threads",
            1,
        )
        frame = read_predictions(pred, read_ids(data / "users.csv"))
        export = pd.read_csv(out / name / "test_predictions.csv.gz")
        pd.testing.assert_frame_equal(
            frame, export[["UserID", *preprocess_targets()]], check_dtype=False
        )
        truth = export[["UserID", *["truth_" + x for x in preprocess_targets()]]].copy()
        truth.columns = ["UserID", *preprocess_targets()]
        official = scores(frame, truth)
        metric = json.loads((out / name / "test_metrics.json").read_text())
        assert official["score"] == pytest.approx(metric["score"])
        assert (Path(str(pred) + ".metadata.json")).is_file()
    week = tmp_path / "week"
    call(
        monkeypatch,
        "evaluate-week",
        "--data",
        data,
        "--models",
        models,
        "--db",
        db,
        "--cohort",
        cohort,
        "--cutoff",
        "2026-01-25T00:00:00+08:00",
        "--out",
        week,
        "--threads",
        1,
    )
    pd.testing.assert_frame_equal(pd.read_csv(week / "summary.csv"), summary)
    bad = json.loads((data / "manifest.json").read_text())
    bad["data_id"] = "wrong-plan"
    (data / "manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="different prepared"):
        call(
            monkeypatch,
            "evaluate",
            "--data",
            data,
            "--models",
            models,
            "--split",
            "test",
            "--out",
            tmp_path / "bad",
        )


def preprocess_targets():
    from relational.data import TARGETS

    return TARGETS


def test_time_encoder_causal_parent_and_gradients():
    t = pd.Timestamp("2026-01-25")
    day = pd.Timedelta(days=1)

    def row(key, kind, age, parent=""):
        return dict(
            key=key,
            kind=kind,
            target=kind,
            time=t - age * day,
            known=t - age * day,
            source=0,
            actor=10,
            repo=42,
            cohort=True,
            parent=parent,
            a0=0.0,
            a1=0.0,
            a2=0.0,
        )

    rows = [row("1:p", 1, 100), row("2:c", 2, 2, "1:p")]
    sample = graph_snapshot(Store(pd.DataFrame(rows), [10]), t, [10], 21)
    future = graph_snapshot(
        Store(pd.DataFrame([*rows, row("5:f", 5, -2)]), [10]), t, [10], 21
    )
    for key in sample:
        if key != "y":
            np.testing.assert_array_equal(sample[key], future[key])
    assert sample["age3"][0] == 100 and sample["x3"][0, 6] == 1
    batch = NeighborSampler(sample).batch(
        np.array([0]), 2, 3, np.random.default_rng(1), "cpu"
    )
    torch.manual_seed(1)
    model = HeteroSAGE(hidden=8, dropout=0)
    original = model(*batch)
    ages = [x.clone() for x in batch[3]]
    ages[4] += 0.25
    assert not torch.allclose(
        original, model(batch[0], batch[1], batch[2], ages), atol=1e-7
    )
    original.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.time_encoders.parameters()
    )
    empty = graph_snapshot(Store(pd.DataFrame(rows), [10]), t + 200 * day, [10], 21)
    eb = NeighborSampler(empty).batch(
        np.array([0]), 2, 3, np.random.default_rng(1), "cpu"
    )
    assert torch.isfinite(model(*eb)).all()
