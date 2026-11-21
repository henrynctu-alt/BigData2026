"""Read the eight official tables without mixing snapshot-local keys."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TARGETS = [
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
]
SPECS = [
    (
        "PullRequests",
        "pr_id",
        "github_pr_id",
        "author_id",
        "created_at",
        "first_seen_at",
    ),
    (
        "Issues",
        "issue_id",
        "github_issue_id",
        "author_id",
        "created_at",
        "first_seen_at",
    ),
    (
        "IssueComments",
        "comment_id",
        "github_comment_id",
        "author_id",
        "created_at",
        "created_at",
    ),
    (
        "PRReviews",
        "review_id",
        "github_review_id",
        "reviewer_id",
        "submitted_at",
        "first_seen_at",
    ),
    (
        "PRReviewComments",
        "comment_id",
        "github_comment_id",
        "author_id",
        "created_at",
        "created_at",
    ),
    (
        "Pushes",
        "push_id",
        "github_push_id",
        "user_id",
        "created_at",
        "created_at",
    ),
]
FK = {
    "PullRequests": {
        "repo_id": "Repositories",
        "author_id": "Users",
        "merged_by_id": "Users",
    },
    "Issues": {
        "repo_id": "Repositories",
        "author_id": "Users",
        "assignee_id": "Users",
        "pull_request_id": "PullRequests",
    },
    "IssueComments": {
        "repo_id": "Repositories",
        "author_id": "Users",
        "issue_id": "Issues",
    },
    "PRReviews": {
        "repo_id": "Repositories",
        "reviewer_id": "Users",
        "pr_id": "PullRequests",
    },
    "PRReviewComments": {
        "repo_id": "Repositories",
        "author_id": "Users",
        "pr_id": "PullRequests",
        "pull_request_review_id": "PRReviews",
        "in_reply_to_id": "PRReviewComments",
    },
    "Pushes": {"repo_id": "Repositories", "user_id": "Users"},
}
PRIMARY = {name: pk for name, pk, *_ in SPECS}
PRIMARY.update(Users="user_id", Repositories="repo_id")


def utc(value):
    """Normalize a calendar cutoff to timezone-naive UTC."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def digest(path):
    """Hash a file in bounded memory."""
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def read_snapshot(path, source, before=None):
    """Resolve every FK inside its own snapshot before combining snapshots."""
    tables = {}
    audit = []
    for name in PRIMARY:
        file = Path(path) / f"{name}.parquet"
        schema = pq.read_schema(file)
        # Strings contain anonymized text, not natural-language semantics.
        # Retain only review state; do not load bodies or future state fields.
        cols = [
            c
            for c in schema.names
            if not (str(schema.field(c).type) == "string" and c not in {"state"})
        ]
        filters = None
        # Keep dimension times for the as-of join, including late endpoints.
        if before is not None and name not in {"Users", "Repositories"}:
            clock = "first_seen_at"
            if name in {"IssueComments", "PRReviewComments", "Pushes"}:
                clock = "created_at"
            filters = [(clock, "<", utc(before).to_pydatetime())]
        tables[name] = pd.read_parquet(file, columns=cols, filters=filters)
        frame = tables[name]
        record = {
            "source": source,
            "table": name,
            "rows": len(frame),
            "bytes": file.stat().st_size,
            "sha256": digest(file),
            "nulls": {c: int(frame[c].isna().sum()) for c in cols},
            "times": {
                c: {"min": str(frame[c].min()), "max": str(frame[c].max())}
                for c in cols
                if c.endswith("_at")
            },
        }
        audit.append(record)
    for record in audit:
        name = record["table"]
        record["foreign_keys"] = {}
        for col, target in FK.get(name, {}).items():
            values = tables[name][col].dropna()
            keys = tables[target][PRIMARY[target]]
            record["foreign_keys"][col] = {
                "non_null": len(values),
                "invalid": int((~values.isin(keys)).sum()),
            }
    users = tables["Users"].set_index("user_id")
    repos = tables["Repositories"].set_index("repo_id")
    parts = []
    for kind, (name, pk, key, actor, event_time, row_time) in enumerate(SPECS):
        raw = tables[name]
        out = pd.DataFrame(
            {
                "key": str(kind) + ":" + raw[key].astype(str),
                "actor": raw[actor]
                .map(users.github_user_id)
                .fillna(-1)
                .astype("int64"),
                "repo": raw.repo_id.map(repos.github_repo_id)
                .fillna(-1)
                .astype("int64"),
                "time": raw[event_time],
                "known": raw[row_time],
                "kind": kind,
                "target": kind,
                "source": source,
            }
        )
        out["known"] = pd.concat(
            [
                out.known,
                raw[actor].map(users.first_seen_at),
                raw.repo_id.map(repos.first_seen_at),
            ],
            axis=1,
        ).max(axis=1)
        out["cohort"] = raw[actor].map(users.is_cohort).fillna(False).astype(bool)
        out["parent"] = ""
        out["a0"] = 0.0
        out["a1"] = 0.0
        out["a2"] = 0.0
        if name == "PullRequests":
            out["a0"] = raw.is_cross_repo.fillna(False).astype(float)
            out["a1"] = raw.draft.fillna(False).astype(float)
        elif name == "Issues":
            is_pr = raw.is_pull_request.fillna(False)
            out.loc[is_pr, "target"] = -1
            out["a0"] = is_pr.astype(float)
            out["a1"] = np.log1p(raw.num_labels.fillna(0).clip(0))
        elif name in {"IssueComments", "PRReviews", "PRReviewComments"}:
            parent_kind = 1 if name == "IssueComments" else 0
            parent_name, parent_pk, parent_key, *_ = SPECS[parent_kind]
            parent_col = "issue_id" if parent_kind == 1 else "pr_id"
            mapping = tables[parent_name].set_index(parent_pk)[parent_key]
            parent = raw[parent_col].map(mapping).astype("Int64")
            out["parent"] = str(parent_kind) + ":" + parent.astype(str)
            out.loc[parent.isna(), "parent"] = ""
            if name == "IssueComments":
                out["a0"] = raw.is_pull_request.fillna(False).astype(float)
            elif name == "PRReviews":
                for j, state in enumerate(
                    ["approved", "commented", "changes_requested"]
                ):
                    out[f"a{j}"] = raw.state.eq(state).astype(float)
            else:
                out["a0"] = raw.in_reply_to_id.notna().astype(float)
        elif name == "Pushes":
            out["a0"] = raw.is_new_ref.fillna(False).astype(float)
        parts.append(out)
    return pd.concat(parts, ignore_index=True), audit


class Store:
    """Versioned row store with explicit point-in-time access."""

    def __init__(self, rows, cohort):
        self.rows = rows.sort_values(["known", "source", "key"])
        self.cohort = np.asarray(cohort, dtype=np.int64)
        self.event_rows = self.rows[
            self.rows.cohort & self.rows.target.ge(0) & self.rows.time.notna()
        ].drop_duplicates("key", keep="first")

    def visible(self, cutoff):
        """Latest available row version; never expose a future row version."""
        cutoff = utc(cutoff)
        return self.rows[self.rows.known < cutoff].drop_duplicates("key", keep="last")

    def events(self, cutoff):
        cutoff = utc(cutoff)
        return self.event_rows[
            self.event_rows.known.lt(cutoff) & self.event_rows.time.lt(cutoff)
        ]

    def labels(self, start, ids, available_until):
        """Return activity in [start, start+7d), known by availability."""
        start, available = utc(start), utc(available_until)
        end = start + pd.Timedelta(days=7)
        if end > available:
            raise ValueError("The seven-day label window is not fully observed.")
        rows = self.event_rows
        events = rows[
            rows.known.lt(available) & rows.time.ge(start) & rows.time.lt(end)
        ]
        counts = events.groupby(["actor", "target"]).size().unstack(fill_value=0)
        return (
            counts.reindex(index=ids, columns=range(6), fill_value=0)
            .gt(0)
            .to_numpy(np.int8)
        )
