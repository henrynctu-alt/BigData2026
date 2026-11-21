"""Causal inputs for the base, temporal and relation-aware encoders."""

import numpy as np
import pandas as pd

from relational.data import utc
from relational.routes import ROUTES

DAY = pd.Timedelta(days=1)
LABELS = 6
EDGE_FIELDS = (
    *(f"peer_count_{i}" for i in range(LABELS)),
    "log_weight",
    "last_age",
    "route_degree",
    "peer_observed",
    "profile_known",
    "unknown_endpoint",
    "recent_peer_trend",
    "edge_share",
    "recent_new_edge",
    "user_repo_focus",
    "bridge_a0",
    "bridge_a1",
    "bridge_a2",
)
DAILY_FIELDS = (
    *(f"count_{i}" for i in range(LABELS)),
    *(f"present_{i}" for i in range(LABELS)),
    *(f"days_since_{i}" for i in range(LABELS)),
    "active",
    "activity_gap",
    "streak",
    "weekday_sin",
    "weekday_cos",
    "known_lag",
    "known_coverage",
    "observed_coverage",
    "review_to_review_comment",
    "issue_to_comment",
    "comment_to_issue",
)
INPUTS = ("base", "daily", "edge", "mask")
SCHEMA_VERSION = 2


def daily_features(events, ids, cutoff, days, coverage=None):
    """Bin by event day using observations available before the cutoff."""
    index = pd.Index(ids)
    start = cutoff - days * DAY
    frame = events.loc[events.actor.isin(ids)]
    user = index.get_indexer(frame.actor)
    day = ((frame.time - start) // DAY).to_numpy(dtype=int)
    counts = np.zeros((len(ids), days, LABELS), np.float32)
    np.add.at(counts, (user, day, frame.target.to_numpy(int)), 1)
    present = counts > 0
    active = present.any(axis=2)
    sequence = np.arange(days)[None, :, None]
    last = np.maximum.accumulate(np.where(present, sequence, -days - 1), 1)
    age = np.minimum(sequence - last, days + 1) / (days + 1)
    active_gap = age.min(axis=2)
    day_index = np.arange(days)[None, :]
    last_idle = np.maximum.accumulate(np.where(~active, day_index, -1), 1)
    streak = (day_index - last_idle) / days
    weekdays = pd.date_range(start + pd.Timedelta(hours=8), periods=days)
    phase = 2 * np.pi * weekdays.dayofweek.to_numpy() / 7
    calendar = np.broadcast_to(
        np.stack([np.sin(phase), np.cos(phase)], axis=1),
        (len(ids), days, 2),
    )
    lag = np.zeros_like(active, dtype=np.float32)
    np.add.at(lag, (user, day), (frame.known - frame.time) / DAY)
    mean_lag = np.log1p(lag.clip(0) / counts.sum(axis=2).clip(1))
    # A missing manifest records unknown coverage, including on zero days.
    quality = np.zeros((days, 2), np.float32)
    if coverage is not None:
        available = (
            coverage[
                (coverage.available_at < cutoff)
                & (coverage.hour_utc >= start)
                & (coverage.hour_utc < cutoff)
            ]
            .sort_values("available_at")
            .drop_duplicates("hour_utc", keep="last")
        )
        buckets = ((available.hour_utc - start) // DAY).to_numpy(int)
        np.add.at(quality[:, 0], buckets, 1 / 24)
        np.add.at(quality[:, 1], buckets, available.observed / 24)
    motifs = []
    for src, dst in [(3, 4), (1, 2), (2, 1)]:
        prior = np.zeros_like(active)
        for shift in (1, 2, 3):
            prior[:, shift:] |= present[:, :-shift, src]
        motifs.append(prior & present[:, :, dst])
    daily = np.concatenate(
        [
            np.log1p(counts),
            present,
            age,
            active[..., None],
            active_gap[..., None],
            streak[..., None],
            calendar,
            mean_lag[..., None],
            np.broadcast_to(quality, (len(ids), days, 2)),
            np.stack(motifs, axis=2),
        ],
        axis=2,
    ).astype(np.float32)
    return counts, daily


def _profile(frame, keys):
    result = frame.groupby([*keys, "target"]).size().unstack(fill_value=0)
    return result.reindex(columns=range(LABELS), fill_value=0)


def graph_features(events, parents, cohort, ids, cutoff, days, cap):
    """Build distinct role-specific endpoints and peer profiles."""
    ids = pd.Index(ids)
    shape = (len(ids), len(ROUTES), cap)
    edges = np.zeros((*shape, len(EDGE_FIELDS)), dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    if events.empty:
        return edges, mask
    ev = events.copy()
    ev["age"] = (cutoff - ev.time) / DAY
    ev["recent"] = ev.age.le(7).astype(int)
    repo_events = ev[ev.repo.ge(0)]
    repo = _profile(repo_events, ["repo"])
    pair = _profile(repo_events, ["actor", "repo"])
    users = _profile(ev, ["actor"])
    repo_recent = repo_events.groupby("repo").recent.sum()
    pair_recent = repo_events.groupby(["actor", "repo"]).recent.sum()
    user_recent = ev.groupby("actor").recent.sum()
    total = users.sum(axis=1)
    pair_total = pair.sum(axis=1)
    shares = pair_total / pair_total.index.get_level_values("actor").map(total)
    focus = shares.pow(2).groupby(level="actor").sum()
    repo_edges = ev.loc[ev.actor.isin(ids)].assign(
        dst=lambda x: x.actor, nbr=lambda x: x.repo, role=lambda x: x.kind
    )
    children = ev[ev.kind.isin([2, 3, 4])].copy()
    children["owner"] = children.parent.map(parents).fillna(-1).astype("int64")
    children = children[children.owner.ne(children.actor)]
    outgoing = children.loc[children.actor.isin(ids)].assign(
        dst=lambda x: x.actor, nbr=lambda x: x.owner, role=lambda x: x.kind + 4
    )
    incoming = children.loc[children.owner.isin(ids)].assign(
        dst=lambda x: x.owner, nbr=lambda x: x.actor, role=lambda x: x.kind + 7
    )
    bridges = pd.concat([repo_edges, outgoing, incoming], ignore_index=True)
    if bridges.empty:
        return edges, mask
    group = bridges.groupby(["dst", "role", "nbr"], sort=True)
    links = group.agg(
        weight=("kind", "size"),
        age=("age", "min"),
        a0=("a0", "mean"),
        a1=("a1", "mean"),
        a2=("a2", "mean"),
    ).reset_index()
    roles = ["dst", "role"]
    links["degree"] = links.groupby(roles).weight.transform("size")
    links["role_share"] = links.weight / links.groupby(roles).weight.transform("sum")
    # Round-robin between 0-3, 3-7 and older days, then strongest endpoint.
    links["bucket"] = np.searchsorted([3, 7], links.age, side="left")
    links = links.sort_values(["weight", "nbr"], ascending=[False, True])
    links["rank"] = links.groupby([*roles, "bucket"]).cumcount()
    links = links.sort_values(["rank", "bucket", "nbr"])
    links["slot"] = links.groupby(roles).cumcount()
    links = links[links.slot < cap]
    for is_repo, part in links.groupby(links.role.lt(6), sort=False):
        nbr = part.nbr.to_numpy()
        known = nbr >= 0
        if is_repo:
            keys = pd.MultiIndex.from_arrays([part.dst, part.nbr])
            own = pair.reindex(keys, fill_value=0).to_numpy()
            peer = repo.reindex(nbr, fill_value=0).to_numpy() - own
            recent_own = pair_recent.reindex(keys, fill_value=0).to_numpy()
            recent = repo_recent.reindex(nbr, fill_value=0).to_numpy() - recent_own
            share = shares.reindex(keys, fill_value=0).to_numpy()
            novel = (recent_own > 0) & (recent_own == own.sum(axis=1))
            # Novelty relative to an earlier subwindow requires >7 days.
            novel &= days > 7
        else:
            # Self-authored parent interactions were already removed.
            peer = users.reindex(nbr, fill_value=0).to_numpy()
            recent = user_recent.reindex(nbr, fill_value=0).to_numpy()
            known = np.isin(nbr, cohort)
            share = part.role_share.to_numpy()
            novel = np.zeros(len(part), bool)
        older = peer.sum(axis=1) - recent
        trend = np.zeros(len(part))
        if days > 7:
            trend = np.log1p(recent / 7) - np.log1p(older / (days - 7))
        values = np.column_stack(
            [
                np.log1p(peer),
                np.log1p(part.weight),
                part.age / days,
                np.log1p(part.degree),
                peer.sum(axis=1) > 0,
                known,
                nbr < 0,
                trend,
                share,
                novel,
                part.dst.map(focus).fillna(0),
                part[["a0", "a1", "a2"]].to_numpy(),
            ]
        )
        ix = (ids.get_indexer(part.dst), part.role.to_numpy(), part.slot)
        edges[ix] = values
        mask[ix] = True
    return edges, mask


def snapshot(store, cutoff, ids, days, cap, coverage=None, labeled=True):
    """Build model inputs and fixed-span baselines, then optional labels."""
    cutoff = utc(cutoff)
    available = store.events(cutoff)
    recent = available[available.time >= cutoff - max(14, days) * DAY]
    events = recent[recent.time >= cutoff - days * DAY]
    counts, daily = daily_features(events, ids, cutoff, days, coverage)
    objects = store.rows[
        store.rows.kind.le(1)
        & store.rows.actor.ge(0)
        & store.rows.known.lt(cutoff)
        & (store.rows.time.isna() | store.rows.time.lt(cutoff))
    ].drop_duplicates("key", keep="last")
    edge, mask = graph_features(
        events,
        objects.set_index("key").actor,
        store.cohort,
        ids,
        cutoff,
        days,
        cap,
    )
    result = dict(
        base=np.log1p(counts.sum(axis=1)),
        daily=daily,
        edge=edge,
        mask=mask,
        ids=np.asarray(ids, np.int64),
        cutoff=np.array(str(cutoff)),
    )
    for span in (7, 14):
        activity = recent[recent.time >= cutoff - span * DAY]
        result[f"p{span}"] = (
            _profile(activity, ["actor"]).reindex(ids, fill_value=0).to_numpy() > 0
        )
    if labeled:
        result["y"] = store.labels(cutoff, ids, cutoff + 7 * DAY)
    return result
