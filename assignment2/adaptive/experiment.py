"""One training/evaluation/inference workflow for the seven retained models."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from relational.data import TARGETS, digest, utc
from relational.evaluation import calibrate, save_prediction

from .dataset import load_sources, new_directory, read_coverage, read_ids
from .features import DAY, INPUTS, SCHEMA_VERSION, snapshot
from .gnn import HeteroSAGE, NeighborSampler, graph_snapshot
from .model import VARIANTS, DualBranchNet, to_device

ALL_MODELS = ("copy7", *VARIANTS, "gnn_time")


class Copy7(torch.nn.Module):
    """Deterministic activity-copy baseline; it has no fitted parameters."""


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check PyTorch build and driver")
    return device


def batches(sample, size, device, shuffle=False, model=None, rng=None):
    if isinstance(model, HeteroSAGE):
        rng = rng if rng is not None else np.random.default_rng(2026)
        sampler = NeighborSampler(sample)
        order = (
            rng.permutation(len(sample["ids"]))
            if shuffle
            else np.arange(len(sample["ids"]))
        )
        for start in range(0, len(order), size):
            rows = order[start : start + size]
            x, edges, roots, ages = sampler.batch(
                rows, model.layers, model.fanout, rng, device
            )
            batch = dict(x=x, edges=edges, roots=roots, ages=ages)
            if shuffle:
                batch["y"] = torch.as_tensor(
                    sample["y"][rows], dtype=torch.float32, device=device
                )
            yield batch
        return
    tensors = {k: torch.from_numpy(sample[k]) for k in INPUTS}
    if shuffle:
        tensors["y"] = torch.from_numpy(sample["y"]).float()
    order = (
        torch.randperm(len(tensors["base"]))
        if shuffle
        else torch.arange(len(tensors["base"]))
    )
    for ix in order.split(size):
        yield to_device({k: v[ix] for k, v in tensors.items()}, device)


@torch.inference_mode()
def probabilities(model, sample, size, device):
    if isinstance(model, Copy7):
        return sample["p7"].astype(np.float32)
    model.eval()
    return np.concatenate(
        [
            model(**b).sigmoid().cpu().numpy()
            for b in batches(sample, size, device, model=model)
        ]
    )


def collect(folder, files, model, size, device):
    chunks = {k: [] for k in ("ids", "weeks", "y", "p7", "p")}
    for file in files:
        with np.load(folder / file) as sample:
            for k in ("ids", "y", "p7"):
                chunks[k].append(sample[k])
            chunks["weeks"].append(np.repeat(str(sample["cutoff"]), len(sample["ids"])))
            chunks["p"].append(probabilities(model, sample, size, device))
    return {k: np.concatenate(v) for k, v in chunks.items()}


def train_stats(folder, files):
    n, sums, squares, positives = 0, np.zeros(6), np.zeros(6), np.zeros(6)
    for name in files:
        with np.load(folder / name) as sample:
            x = sample["base"].astype(np.float64)
            n += len(x)
            sums += x.sum(axis=0)
            squares += np.square(x).sum(axis=0)
            positives += sample["y"].sum(axis=0)
    mean = sums / n
    std = np.sqrt(np.maximum(squares / n - mean**2, 0))
    std[std < 1e-6] = 1
    return [
        torch.as_tensor(x, dtype=torch.float32)
        for x in (mean, std, np.clip(positives / n, 1e-4, 1 - 1e-4))
    ]


def manifest(folder):
    meta = json.loads((folder / "manifest.json").read_text())
    if meta["schema"] != SCHEMA_VERSION:
        raise ValueError("Rerun prepare: this submission requires unified schema 2")
    if digest(folder / "users.csv") != meta["ids_sha256"]:
        raise ValueError("Prepared user list has changed")
    return meta


def build_model(cfg):
    if cfg["variant"] == "copy7":
        return Copy7()
    if cfg["variant"] == "gnn_time":
        model = HeteroSAGE(
            hidden=cfg["hidden"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
            time_encoding="relative",
            time_dim=cfg["time_dim"],
            time_periods=cfg["time_periods"],
            history_days=cfg["days"],
        )
        model.fanout = cfg["fanout"]
        return model
    return DualBranchNet(**cfg)


def train(args):
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    meta = manifest(args.data)
    mean, std, prev = train_stats(args.data, meta["files"]["train"])
    weight = (
        ((1 - prev) / prev).sqrt().clamp(1, 12).to(device)
        if args.loss == "weighted"
        else None
    )
    with new_directory(args.out) as out:
        for variant in args.models:
            torch.manual_seed(args.seed)
            rng = np.random.default_rng(args.seed)
            cfg = dict(variant=variant, days=meta["plan"]["history_days"])
            if variant == "gnn_time":
                cfg.update(
                    hidden=args.gnn_hidden,
                    layers=args.gnn_layers,
                    fanout=args.gnn_fanout,
                    dropout=args.gnn_dropout,
                    time_dim=args.time_dim,
                    time_periods=args.time_periods,
                )
            elif variant != "copy7":
                cfg.update(hidden=args.hidden, branch_dropout=args.branch_dropout)
            model = build_model(cfg).to(device)
            history, best_epoch = [], 0
            if variant != "copy7":
                with torch.no_grad():
                    if variant == "gnn_time":
                        model.head.bias.copy_(prev.logit())
                    else:
                        model.mean.copy_(mean)
                        model.std.copy_(std)
                        model.prior.bias.copy_(prev.logit())
                optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
                best = float("inf")
                for epoch in range(1, args.epochs + 1):
                    model.train()
                    loss_sum, n = 0.0, 0
                    for name in rng.permutation(meta["files"]["train"]):
                        with np.load(args.data / name) as sample:
                            for batch in batches(
                                sample, args.batch_size, device, True, model, rng
                            ):
                                truth = batch.pop("y")
                                optim.zero_grad(set_to_none=True)
                                loss = F.binary_cross_entropy_with_logits(
                                    model(**batch), truth, pos_weight=weight
                                )
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 5, error_if_nonfinite=True
                                )
                                optim.step()
                                loss_sum += loss.item() * len(truth)
                                n += len(truth)
                    stop = collect(
                        args.data, meta["files"]["stop"], model, args.batch_size, device
                    )
                    p = stop["p"].astype(np.float64).clip(1e-12, 1 - 1e-12)
                    y = stop["y"]
                    val = float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())
                    history.append(dict(epoch=epoch, train=loss_sum / n, stop=val))
                    print(variant, history[-1], flush=True)
                    if val < best:
                        best, best_epoch = val, epoch
                        state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                    if epoch - best_epoch >= args.patience:
                        break
                model.load_state_dict(state)
                cal = collect(
                    args.data,
                    meta["files"]["calibration"],
                    model,
                    args.batch_size,
                    device,
                )
                thresholds = calibrate(cal["y"], cal["p"]).tolist()
            else:
                thresholds = [0.5] * 6
            record = dict(
                schema=SCHEMA_VERSION,
                data_id=meta["data_id"],
                model=cfg,
                targets=TARGETS,
                neighbors=meta["plan"]["neighbors"],
                thresholds=thresholds,
                best_epoch=best_epoch,
                opt=args.loss,
                params=sum(p.numel() for p in model.parameters()),
                seed=args.seed,
                torch_version=str(torch.__version__),
                device=str(device),
                batch_size=args.batch_size,
                lr=args.lr,
                max_epochs=args.epochs,
                patience=args.patience,
                coverage_used=meta["coverage_sha256"] is not None,
                coverage_sha256=meta["coverage_sha256"],
                cohort_sha256=meta["cohort_sha256"],
                available_from_utc=str(utc(meta["plan"]["calibration"][-1]) + 7 * DAY),
                splits=meta["files"],
                sources=meta["sources"],
            )
            directory = out / variant
            directory.mkdir()
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({**record, "state": state}, directory / "model.pt")
            (directory / "run.json").write_text(json.dumps(record, indent=2))
            pd.DataFrame(history, columns=["epoch", "train", "stop"]).to_csv(
                directory / "history.csv", index=False
            )
        (out / "environment.json").write_text(
            json.dumps(
                dict(
                    python=platform.python_version(),
                    torch=str(torch.__version__),
                    numpy=np.__version__,
                    pandas=pd.__version__,
                ),
                indent=2,
            )
        )


def restore(path, device):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["schema"] != SCHEMA_VERSION or record["targets"] != TARGETS:
        raise ValueError(
            "Checkpoint schema/target mismatch; retrain with this submission"
        )
    model = build_model(record["model"])
    model.load_state_dict(record["state"])
    return model.to(device).eval(), record


def validate_inputs(record, cohort, coverage):
    if digest(cohort) != record["cohort_sha256"]:
        raise ValueError("Cohort differs from training")
    if record["coverage_used"] != (coverage is not None):
        raise ValueError("Use the same coverage policy as training")


def export_result(folder, split, model_name, data, record, tables):
    directory = folder / model_name
    directory.mkdir()
    result = save_prediction(
        directory, split, data, data["p"], np.asarray(record["thresholds"])
    )
    tables["summary"].append(
        dict(
            model=model_name,
            n=len(data["y"]),
            params=record["params"],
            **{k: result[k] for k in ("macro_f1", "micro_f1", "score")},
        )
    )
    for cutoff, item in [("pooled", result), *result["weeks"].items()]:
        tables["weekly"].append(
            dict(
                model=model_name,
                cutoff_utc=cutoff,
                **{k: item[k] for k in ("macro_f1", "micro_f1", "score")},
            )
        )
    from sklearn.metrics import average_precision_score

    for week in [None, *np.unique(data["weeks"])]:
        take = np.ones(len(data["y"]), bool) if week is None else data["weeks"] == week
        y, p = data["y"][take], data["p"][take]
        from relational.evaluation import metrics

        m = metrics(y, p >= np.asarray(record["thresholds"]))
        for j, label in enumerate(TARGETS):
            tables["per_label"].append(
                dict(
                    model=model_name,
                    cutoff_utc="pooled" if week is None else week,
                    label=label,
                    f1=m["f1"][j],
                    tp=m["tp"][j],
                    fp=m["fp"][j],
                    fn=m["fn"][j],
                    support=m["support"][j],
                    ap=float(average_precision_score(y[:, j], p[:, j]))
                    if y[:, j].sum()
                    else None,
                )
            )


def evaluate(args):
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    meta = manifest(args.data)
    paths = sorted(args.models.glob("*/model.pt"))
    if not paths:
        raise ValueError("No checkpoints found")
    week = getattr(args, "cutoff", None)
    store, ids, coverage = None, None, None
    if week:
        t = utc(week)
        if t != t.floor("h") or t + 7 * DAY > utc(meta["plan"]["observation_end"]):
            raise ValueError(
                "Weekly label window is outside the plan observation bounds"
            )
        ids = read_ids(args.data / "users.csv")
        store, audits = load_sources(args.db, read_ids(args.cohort), t + 7 * DAY)
        if [[a["source"], a["table"], a["sha256"]] for a in audits] != meta["sources"]:
            raise ValueError("Use original source tables in order for evaluate-week")
        coverage = read_coverage(args.coverage)
    tables = {k: [] for k in ("summary", "weekly", "per_label")}
    with new_directory(args.out) as out:
        sample = None
        for path in paths:
            model, record = restore(path, device)
            if record["data_id"] != meta["data_id"]:
                raise ValueError("Checkpoint trained with a different prepared dataset")
            size = args.batch_size or record["batch_size"]
            if week:
                validate_inputs(record, args.cohort, args.coverage)
                if t < utc(record["available_from_utc"]):
                    raise ValueError("Cutoff precedes calibration")
                if sample is None:
                    sample = snapshot(
                        store,
                        t,
                        ids,
                        meta["plan"]["history_days"],
                        meta["plan"]["neighbors"],
                        coverage,
                    )
                    sample.update(
                        graph_snapshot(store, t, ids, meta["plan"]["history_days"])
                    )
                data = dict(
                    ids=ids,
                    weeks=np.repeat(str(t), len(ids)),
                    y=sample["y"],
                    p=probabilities(model, sample, size, device),
                )
            else:
                data = collect(
                    args.data, meta["files"][args.split], model, size, device
                )
            export_result(
                out,
                "test" if week else args.split,
                path.parent.name,
                data,
                record,
                tables,
            )
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
        print(pd.DataFrame(tables["summary"]).to_string(index=False))


def predict(args):
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    model, record = restore(args.checkpoint, device)
    validate_inputs(record, args.cohort, args.coverage)
    ids, cohort = read_ids(args.users), read_ids(args.cohort)
    if not np.isin(ids, cohort).all():
        raise ValueError("Requested users must belong to the cohort")
    t = utc(args.cutoff)
    if t != t.floor("h") or t < utc(record["available_from_utc"]):
        raise ValueError("Use an hourly cutoff after calibration labels close")
    if args.out.exists() or Path(str(args.out) + ".metadata.json").exists():
        raise FileExistsError("Prediction output already exists")
    store, audit = load_sources(args.db, cohort, t)
    if isinstance(model, HeteroSAGE):
        sample = graph_snapshot(store, t, ids, record["model"]["days"], labeled=False)
    else:
        sample = snapshot(
            store,
            t,
            ids,
            record["model"]["days"],
            record["neighbors"],
            read_coverage(args.coverage),
            labeled=False,
        )
    p = probabilities(model, sample, args.batch_size or record["batch_size"], device)
    result = pd.DataFrame(
        (p >= np.asarray(record["thresholds"])).astype(int), columns=TARGETS
    )
    result.insert(0, "UserID", ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, mode="x")
    from official.scripts.evaluate import read_predictions

    read_predictions(args.out, ids)
    metadata = dict(
        model=record["model"],
        cutoff_utc=str(t),
        label_end_utc=str(t + 7 * DAY),
        checkpoint_sha256=digest(args.checkpoint),
        users_sha256=digest(args.users),
        thresholds=record["thresholds"],
        batch_size=args.batch_size or record["batch_size"],
        sources=[[a["source"], a["table"], a["sha256"]] for a in audit],
        source_known_max=str(store.rows.known.max()),
        python=platform.python_version(),
        torch=str(torch.__version__),
    )
    Path(str(args.out) + ".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {args.out}: {len(ids)} users, official format checked")
