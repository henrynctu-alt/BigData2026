"""EDA of the official eight tables, bounded by the plan's validation end."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from adaptive.dataset import calendar, load_sources, new_directory, read_ids
from adaptive.features import DAY
from relational.data import TARGETS, utc
from relational.evaluation import metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, nargs="+", required=True)
    p.add_argument("--plan", type=Path, default=Path("configs/historical.json"))
    p.add_argument("--cohort", type=Path, default=Path("official/scripts/cohort.txt"))
    p.add_argument("--users", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    dates = calendar(plan)
    end = dates["evaluation"][-1] + 7 * DAY
    cohort = read_ids(a.cohort)
    ids = read_ids(a.users) if a.users else cohort
    if not np.isin(ids, cohort).all():
        raise ValueError("EDA users outside cohort")
    store, audits = load_sources(a.db, cohort, plan["observation_end"])
    start = utc(plan["observation_start"])
    ev = store.events(end)
    ev = ev[ev.time.ge(start)].copy()
    with new_directory(a.out) as out:
        (out / "input_audit.json").write_text(json.dumps(audits, indent=2))
        daily = (
            ev.groupby([ev.time.dt.floor("D"), "target"]).size().unstack(fill_value=0)
        )
        daily = daily.reindex(
            index=pd.date_range(
                start.floor("D"), end.floor("D"), freq="D", inclusive="left"
            ),
            columns=range(6),
            fill_value=0,
        )
        daily.columns = TARGETS
        daily.to_csv(out / "daily_events_train_validation.csv", index_label="utc_day")
        hours = ev.groupby([ev.time.dt.hour, "target"]).size().unstack(fill_value=0)
        hours = hours.reindex(index=range(24), columns=range(6), fill_value=0)
        hours.columns = TARGETS
        hours.to_csv(out / "hour_of_day_counts.csv", index_label="utc_hour")
        weekly, labels = [], []
        cutoffs = [
            t
            for split in ("train", "stop", "calibration", "evaluation")
            for t in dates[split]
        ]
        for t in cutoffs:
            y = store.labels(t, ids, t + 7 * DAY)
            past = store.events(t)
            past = past[past.time.ge(t - 7 * DAY)]
            counts = past.groupby(["actor", "target"]).size().unstack(fill_value=0)
            copy = (
                counts.reindex(index=ids, columns=range(6), fill_value=0).to_numpy() > 0
            )
            row = dict(cutoff_utc=str(t), score_copy7=metrics(y, copy)["score"])
            row.update({k: float(y[:, j].mean()) for j, k in enumerate(TARGETS)})
            weekly.append(row)
            labels.append(y)
        weeks = pd.DataFrame(weekly)
        weeks.to_csv(out / "weekly_prevalence_and_copy.csv", index=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(np.concatenate(labels).T)
        pd.DataFrame(corr, index=TARGETS, columns=TARGETS).to_csv(
            out / "label_correlation.csv"
        )
        recurrence = []
        previous, following = [], []
        for i in range(len(cutoffs) - 1):
            if cutoffs[i + 1] - cutoffs[i] == 7 * DAY:
                previous.append(labels[i])
                following.append(labels[i + 1])
        if previous:
            before, after = np.concatenate(previous), np.concatenate(following)
            for j, label in enumerate(TARGETS):
                recurrence.append(
                    dict(
                        target=label,
                        p_next_given_active=float(after[before[:, j] == 1, j].mean())
                        if before[:, j].any()
                        else None,
                        p_next_given_inactive=float(after[before[:, j] == 0, j].mean())
                        if (before[:, j] == 0).any()
                        else None,
                    )
                )
        pd.DataFrame(recurrence).to_csv(out / "activity_recurrence.csv", index=False)
        # Cross-table connectivity of recent cohort activity, before validation ends.
        recent = ev[ev.time.ge(end - plan["history_days"] * DAY)]
        pairs = (
            recent[recent.repo.ge(0)].groupby(["actor", "repo"]).size().rename("events")
        )
        pairs.reset_index().to_csv(out / "user_repository_activity.csv", index=False)
        repo_degree = pairs.reset_index().groupby("repo").actor.nunique()
        user_degree = pairs.reset_index().groupby("actor").repo.nunique()
        delays = []
        rows = store.rows[store.rows.known.lt(end)]
        lag = (rows.known - rows.time) / DAY
        for kind, label in enumerate(TARGETS):
            values = lag[rows.kind.eq(kind)].dropna()
            delays.append(
                dict(
                    target=label,
                    n=len(values),
                    median_days=float(values.median()) if len(values) else None,
                    p95_days=float(values.quantile(0.95)) if len(values) else None,
                )
            )
        pd.DataFrame(delays).to_csv(out / "row_availability_delay.csv", index=False)
        summary = dict(
            eda_start_utc=str(start),
            eda_end_exclusive_utc=str(end),
            n_users=len(ids),
            n_cohort=len(cohort),
            observed_events=len(ev),
            history_days=plan["history_days"],
            observed_event_time_min=str(ev.time.min()),
            observed_event_time_max=str(ev.time.max()),
            recent_repo_degree_quantiles=repo_degree.quantile(
                [0.5, 0.9, 0.99, 1]
            ).to_dict(),
            recent_user_degree_quantiles=user_degree.quantile(
                [0.5, 0.9, 0.99, 1]
            ).to_dict(),
            note=(
                "Test tables are audited; target EDA excludes test weeks. "
                "Zero events do not prove complete source coverage."
            ),
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        daily.rolling(7, min_periods=1).mean().plot(ax=axes[0])
        axes[0].set(
            title="Observed event counts: seven-day mean",
            ylabel="Events",
            xlabel="UTC day",
        )
        weeks.set_index("cutoff_utc")[TARGETS].plot(ax=axes[1])
        axes[1].set(
            title="Weekly target prevalence before test",
            ylabel="Positive fraction",
            xlabel="UTC cutoff",
        )
        fig.tight_layout()
        fig.savefig(out / "temporal_drift.png", dpi=160)
        plt.close(fig)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
