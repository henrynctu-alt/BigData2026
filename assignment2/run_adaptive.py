"""Unified CLI: prepare, train, evaluate, weekly ablation and predict."""

import argparse
from pathlib import Path

import numpy as np

from adaptive.dataset import prepare
from adaptive.experiment import ALL_MODELS, evaluate, predict, train


def model_name(value):
    return {"addaptive": "adaptive", "gnn_relative": "gnn_time"}.get(value, value)


def runtime(parser, training=False):
    parser.add_argument("--device", "--to-device", "--to_device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256 if training else None)
    parser.add_argument("--threads", type=int, default=2)


def source(parser):
    parser.add_argument("--db", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--cohort", type=Path, default=Path("official/scripts/cohort.txt")
    )
    parser.add_argument("--coverage", type=Path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare", help="Create shared adaptive/GNN cutoff caches")
    source(prep)
    prep.add_argument("--plan", type=Path, required=True)
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--users", type=Path)
    prep.add_argument("--sample-users", type=int)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(run=prepare)
    fit = stages.add_parser("train", help="Train all seven retained models")
    runtime(fit, True)
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument(
        "--models",
        type=model_name,
        nargs="+",
        choices=ALL_MODELS,
        default=list(ALL_MODELS),
    )
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--epochs", type=int, default=16)
    fit.add_argument("--patience", type=int, default=4)
    fit.add_argument("--hidden", type=int, default=32)
    fit.add_argument("--lr", type=float, default=0.0005)
    fit.add_argument("--loss", choices=["weighted", "bce"], default="weighted")
    fit.add_argument("--branch-dropout", type=float, default=0.1)
    fit.add_argument("--gnn-hidden", type=int, default=32)
    fit.add_argument("--gnn-layers", type=int, default=2)
    fit.add_argument("--gnn-fanout", type=int, default=8)
    fit.add_argument("--gnn-dropout", type=float, default=0.1)
    fit.add_argument("--time-dim", type=int, default=16)
    fit.add_argument("--time-periods", type=float, nargs="+", default=[1.0, 7.0, 28.0])
    fit.set_defaults(run=train)
    for name in ("evaluate", "evaluate-week"):
        score = stages.add_parser(name)
        runtime(score)
        score.add_argument("--data", type=Path, required=True)
        score.add_argument("--models", type=Path, required=True)
        score.add_argument("--out", type=Path, required=True)
        if name == "evaluate":
            score.add_argument(
                "--split", choices=["evaluation", "test"], default="evaluation"
            )
        else:
            source(score)
            score.add_argument("--cutoff", required=True)
        score.set_defaults(run=evaluate)
    pred = stages.add_parser("predict", help="Write official seven-column output CSV")
    runtime(pred)
    source(pred)
    pred.add_argument("--checkpoint", type=Path, required=True)
    pred.add_argument("--cutoff", required=True)
    pred.add_argument("--users", type=Path, required=True)
    pred.add_argument("--out", type=Path, required=True)
    pred.set_defaults(run=predict)
    args = p.parse_args()
    for key in (
        "batch_size",
        "threads",
        "epochs",
        "patience",
        "hidden",
        "gnn_hidden",
        "gnn_layers",
        "gnn_fanout",
        "time_dim",
    ):
        value = getattr(args, key, None)
        if value is not None and value < 1:
            p.error(f"{key} must be positive")
    if args.stage == "train":
        if not np.isfinite(args.lr) or args.lr <= 0:
            p.error("lr must be finite and positive")
        if any(not 0 <= v < 1 for v in (args.branch_dropout, args.gnn_dropout)):
            p.error("dropout must be in [0,1)")
        if any(not np.isfinite(v) or v <= 0 for v in args.time_periods):
            p.error("time-periods must be positive finite days")
        if len(args.models) != len(set(args.models)):
            p.error("Duplicate model")
    args.run(args)


if __name__ == "__main__":
    main()
