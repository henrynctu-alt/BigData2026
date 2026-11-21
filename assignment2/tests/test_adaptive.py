"""Point-in-time, peer-message, ablation and device regression tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from adaptive.dataset import calendar
from adaptive.features import DAILY_FIELDS, EDGE_FIELDS, INPUTS, snapshot
from adaptive.model import DualBranchNet, to_device
from relational.data import Store

T = pd.Timestamp("2026-01-25")
DAY = pd.Timedelta(days=1)
IDS = [10, 20, 30]


def event(key, actor, kind, day, *, lag=0, repo=101, parent=""):
    return dict(
        key=key,
        actor=actor,
        kind=kind,
        target=kind,
        repo=repo,
        source=0,
        time=T + day * DAY,
        known=T + (day + lag) * DAY,
        parent=parent,
        cohort=actor in IDS,
        a0=0.0,
        a1=0.0,
        a2=0.0,
    )


def snap(rows, **kwargs):
    return snapshot(Store(pd.DataFrame(rows), IDS), T, IDS, 21, 4, **kwargs)


def test_event_and_observation_boundaries():
    rows = [
        event("a", 10, 0, -21),
        event("b", 10, 2, -1, lag=1),
        event("c", 10, 5, 0),
        event("d", 10, 3, 6, lag=1),
        event("e", 10, 4, 7),
    ]
    result = snap(rows)
    np.testing.assert_allclose(np.expm1(result["base"][0]), [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(result["y"][0], [0, 0, 0, 0, 0, 1])
    older = snap(rows + [event("later", 20, 1, -2, lag=4)])
    for k in INPUTS:
        np.testing.assert_array_equal(older[k], result[k])
    assert "y" not in snap(rows, labeled=False)


def test_distinct_neighbors_leave_one_out_and_unknown_parent():
    rows = [
        event("p1", 10, 5, -8),
        event("p2", 10, 5, -2),
        event("p3", 20, 2, -3, parent="unknown"),
        event("p4", 10, 3, -2, parent="pr"),
        event("pr", 30, 0, -5, lag=5),
    ]
    result = snap(rows)
    assert result["mask"][0, 5].sum() == 1
    profile = result["edge"][0, 5, 0]
    np.testing.assert_allclose(np.expm1(profile[:6]), [0, 0, 1, 0, 0, 0])
    assert np.expm1(profile[EDGE_FIELDS.index("log_weight")]) == pytest.approx(2)
    assert result["edge"][0, 7, 0, EDGE_FIELDS.index("unknown_endpoint")] == 1
    assert not result["mask"][2].any()
    rows[-1]["known"] = T - DAY
    known = snap(rows)
    assert known["edge"][0, 7, 0, EDGE_FIELDS.index("profile_known")] == 1
    assert known["mask"][2, 10, 0]


def test_temporal_order_and_manifest_availability():
    rows = [event("1", 10, 3, -4), event("2", 10, 4, -2)]
    coverage = pd.DataFrame(
        dict(
            hour_utc=[T - DAY, T - DAY],
            available_at=[T - DAY, T],
            observed=[0, 1],
        )
    )
    first = snap(rows, coverage=coverage)
    reverse = snap([event("1", 10, 3, -2), event("2", 10, 4, -4)])
    np.testing.assert_array_equal(first["base"], reverse["base"])
    motif = DAILY_FIELDS.index("review_to_review_comment")
    assert first["daily"][0, :, motif].sum() == 1
    assert reverse["daily"][0, :, motif].sum() == 0
    assert first["daily"][0, -1, DAILY_FIELDS.index("known_coverage")] == (
        pytest.approx(1 / 24)
    )
    assert first["daily"][0, -1, DAILY_FIELDS.index("observed_coverage")] == 0


def test_seven_day_encoder_still_has_fourteen_day_baseline():
    store = Store(pd.DataFrame([event("1", 10, 5, -10)]), IDS)
    result = snapshot(store, T, IDS, 7, 4)
    assert not result["p7"].any()
    assert result["p14"][0, 5]
    assert not result["base"].any()
    assert not result["mask"].any()
    empty = snapshot(store, T + 100 * DAY, IDS, 7, 4)
    assert not empty["p14"].any()
    assert np.isfinite(empty["daily"]).all()


@pytest.mark.parametrize("variant", ["base", "graph", "temporal"])
def test_disabled_inputs_do_not_change_logits(variant):
    sample = snap([event("a", 10, 5, -2), event("b", 20, 2, -1)])
    inputs = {k: torch.from_numpy(sample[k]) for k in INPUTS}
    model = DualBranchNet(variant, 21, 8).eval()
    expected = model(**inputs)
    if variant in ("base", "temporal"):
        inputs["edge"] = torch.randn_like(inputs["edge"])
        inputs["mask"] = ~inputs["mask"]
    if variant in ("base", "graph"):
        inputs["daily"] = torch.randn_like(inputs["daily"])
    torch.testing.assert_close(model(**inputs), expected)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="No CUDA device"
            ),
        ),
    ],
)
def test_branch_gradients_cold_start_and_device_roundtrip(device):
    sample = snap([event("a", 10, 5, -2), event("b", 20, 2, -1), event("c", 10, 3, 0)])
    inputs = to_device({k: torch.from_numpy(sample[k]) for k in INPUTS}, device)
    torch.manual_seed(23)
    model = DualBranchNet("adaptive", 21, 8, 0).to(device)
    logits = model(**inputs)
    torch.testing.assert_close(logits[2], model.prior(model.base(inputs["base"][2])))
    target = torch.from_numpy(sample["y"]).float().to(device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    for module in (
        model.graph,
        model.temporal,
        model.gates,
        *model.heads.values(),
    ):
        assert any(
            p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
    torch.optim.AdamW(model.parameters()).step()
    model.eval()
    result = model(**inputs).detach().cpu()
    cpu = model.to("cpu")(**to_device(inputs, "cpu"))
    torch.testing.assert_close(cpu, result, rtol=1e-4, atol=1e-5)


def test_plan_bounds_and_split_overlap():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/historical.json").read_text())
    assert len(calendar(plan)["test"]) == 6
    plan["stop"] = [plan["train"]["last"]]
    with pytest.raises(ValueError, match="Overlapping"):
        calendar(plan)
    plan["stop"] = ["2026-08-02T00:00:00+08:00"]
    plan["observation_end"] = plan["test"][-1]
    with pytest.raises(ValueError, match="bounds"):
        calendar(plan)
