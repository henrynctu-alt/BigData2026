"""Compare exports only after matching user/cutoff/truth keys."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from relational.data import TARGETS
from relational.evaluation import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--split", choices=("evaluation", "test"), default="test")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    reference, rows = None, []
    truth_columns = ["truth_" + x for x in TARGETS]
    for run in args.runs:
        paths = sorted(run.glob(f"*/{args.split}_predictions.csv.gz"))
        if not paths:
            raise ValueError(f"No prediction files in {run}")
        for path in paths:
            frame = pd.read_csv(path)
            frame["cutoff_utc"] = pd.to_datetime(frame.cutoff_utc, utc=True)
            frame = frame.set_index(["cutoff_utc", "UserID"]).sort_index()
            if not frame.index.is_unique:
                raise ValueError(f"Duplicate user-week: {path}")
            truth = frame[truth_columns].to_numpy(bool)
            if reference is None:
                reference = (frame.index, truth)
            elif not (
                reference[0].equals(frame.index) and np.array_equal(reference[1], truth)
            ):
                raise ValueError(f"User/cutoff/ground-truth mismatch: {path}")
            for cutoff in [None, *frame.index.get_level_values(0).unique()]:
                take = (
                    np.ones(len(frame), bool)
                    if cutoff is None
                    else frame.index.get_level_values(0) == cutoff
                )
                result = metrics(truth[take], frame[TARGETS].to_numpy()[take])
                rows.append(
                    dict(
                        run=str(run),
                        model=path.parent.name,
                        cutoff_utc="pooled" if cutoff is None else str(cutoff),
                        n=int(take.sum()),
                        **{k: result[k] for k in ("macro_f1", "micro_f1", "score")},
                    )
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        pd.DataFrame(rows).to_csv(handle, index=False)
    print(pd.DataFrame(rows).query("cutoff_utc == 'pooled'").to_string(index=False))


if __name__ == "__main__":
    main()
