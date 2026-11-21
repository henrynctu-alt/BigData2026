"""Table-row hetero GraphSAGE, adapted from the RelBench GNN tutorial.

Pure PyTorch implementation: typed feature encoders, relation-specific mean
aggregation, reverse FK edges, and bounded incoming neighbor sampling.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from adaptive.features import DAY
from relational.data import utc

TYPES = (
    "Users",
    "Repositories",
    "PullRequests",
    "Issues",
    "IssueComments",
    "PRReviews",
    "PRReviewComments",
    "Pushes",
)
RELATIONS = []
for _kind in range(6):
    _table = _kind + 2
    for _source in (0, 1):
        RELATIONS.extend([(_source, _table), (_table, _source)])
for _child, _parent in ((4, 3), (5, 2), (6, 2)):
    RELATIONS.extend([(_parent, _child), (_child, _parent)])
FIELDS = 10  # six log counts, age/history, three bridge attributes


def graph_snapshot(store, cutoff, ids, days, labeled=True):
    """Build a cutoff graph from visible row versions and canonical FK IDs.

    Dynamic rows are in [t-H,t). Older issue/PR rows may be retained solely
    as already-known parent endpoints. Their author/repo links are retained.
    No mutable close/merge fields or learned ID embeddings are used.
    """
    t = utc(cutoff)
    visible = store.visible(t)
    visible = visible[visible.time.isna() | visible.time.lt(t)]
    dynamic = visible[visible.time.ge(t - days * DAY)]
    parents = visible[visible.key.isin(dynamic.parent) & visible.kind.le(1)]
    rows = pd.concat([dynamic, parents]).drop_duplicates("key")
    tables = [rows[rows.kind.eq(k)].sort_values("key") for k in range(6)]
    users = np.unique(np.concatenate([np.asarray(ids), rows.actor.to_numpy()]))
    users = users[users >= 0]
    repos = np.sort(rows.loc[rows.repo.ge(0), "repo"].unique())
    indices = [pd.Index(users), pd.Index(repos)] + [pd.Index(f.key) for f in tables]
    xs = [np.zeros((len(ix), FIELDS), np.float32) for ix in indices]
    events = store.events(t)
    events = events[events.time.ge(t - days * DAY)]
    for typ, key in ((0, "actor"), (1, "repo")):
        profile = events.groupby([key, "target"]).size().unstack(fill_value=0)
        xs[typ][:, :6] = np.log1p(
            profile.reindex(
                index=indices[typ], columns=range(6), fill_value=0
            ).to_numpy()
        )
    ages = [np.zeros(len(ix), np.float32) for ix in indices]
    edges = []
    for kind, frame in enumerate(tables):
        typ = kind + 2
        ages[typ] = (
            ((t - frame.time) / DAY).fillna(days).clip(lower=0).to_numpy(np.float32)
        )
        xs[typ][:, 6] = np.minimum(ages[typ], days) / days
        xs[typ][:, 7:] = frame[["a0", "a1", "a2"]].fillna(0).to_numpy()
        for src, key in ((0, "actor"), (1, "repo")):
            origin = indices[src].get_indexer(frame[key])
            dest = np.arange(len(frame))
            valid = origin >= 0
            edge = np.stack([origin[valid], dest[valid]])
            edges.extend([edge, edge[::-1].copy()])
    for child, parent in ((4, 3), (5, 2), (6, 2)):
        frame = tables[child - 2]
        origin = indices[parent].get_indexer(frame.parent)
        valid = origin >= 0
        edge = np.stack([origin[valid], np.arange(len(frame))[valid]])
        edges.extend([edge, edge[::-1].copy()])
    sample = {f"x{i}": x for i, x in enumerate(xs)}
    sample.update({f"e{i}": e.astype(np.int64) for i, e in enumerate(edges)})
    sample.update({f"age{i}": age for i, age in enumerate(ages)})
    sample["history_days"] = np.array(days)
    sample.update(
        ids=np.asarray(ids), seeds=indices[0].get_indexer(ids), cutoff=np.array(str(t))
    )
    for span in (7, 14):
        recent = store.events(t)
        recent = recent[recent.time.ge(t - span * DAY)]
        counts = recent.groupby(["actor", "target"]).size().unstack(fill_value=0)
        sample[f"p{span}"] = (
            counts.reindex(index=ids, columns=range(6), fill_value=0).to_numpy() > 0
        )
    if labeled:
        sample["y"] = store.labels(t, ids, t + 7 * DAY)
    return sample


class NeighborSampler:
    """Sample at most fanout incoming edges per destination and relation."""

    def __init__(self, sample):
        self.sample = sample
        self.csr = []
        for r, (_, dst) in enumerate(RELATIONS):
            edge = sample[f"e{r}"]
            order = np.argsort(edge[1], kind="stable")
            source = edge[0, order]
            counts = np.bincount(edge[1], minlength=len(sample[f"x{dst}"]))
            self.csr.append((source, np.r_[0, counts.cumsum()]))

    def batch(self, rows, layers, fanout, rng, device):
        seeds = self.sample["seeds"][rows]
        chosen = [set() for _ in TYPES]
        chosen[0].update(seeds.tolist())
        frontier = [set(s) for s in chosen]
        selected = [set() for _ in RELATIONS]
        for _ in range(layers):
            following = [set() for _ in TYPES]
            for r, (src, dst) in enumerate(RELATIONS):
                source, ptr = self.csr[r]
                for node in sorted(frontier[dst]):
                    neighbors = source[ptr[node] : ptr[node + 1]]
                    if len(neighbors) > fanout:
                        neighbors = rng.choice(neighbors, fanout, replace=False)
                    for neighbor in neighbors.tolist():
                        selected[r].add((neighbor, node))
                        if neighbor not in chosen[src]:
                            following[src].add(neighbor)
            for i in range(len(TYPES)):
                chosen[i].update(following[i])
            frontier = following
        nodes = [np.asarray(sorted(s), dtype=np.int64) for s in chosen]
        x = [
            torch.as_tensor(self.sample[f"x{i}"][n], device=device)
            for i, n in enumerate(nodes)
        ]
        edges = []
        for r, (src, dst) in enumerate(RELATIONS):
            raw = np.asarray(sorted(selected[r]), dtype=np.int64).reshape(-1, 2)
            edge = np.stack(
                [
                    np.searchsorted(nodes[src], raw[:, 0]),
                    np.searchsorted(nodes[dst], raw[:, 1]),
                ]
            )
            edges.append(torch.as_tensor(edge, device=device))
        root = torch.as_tensor(np.searchsorted(nodes[0], seeds), device=device)
        ages = None
        if "age0" in self.sample:
            ages = [
                torch.as_tensor(self.sample[f"age{i}"][n], device=device)
                for i, n in enumerate(nodes)
            ]
        return x, edges, root, ages


class HeteroSAGE(nn.Module):
    """Per-type encoders followed by relation-specific GraphSAGE layers."""

    def __init__(
        self,
        hidden=64,
        layers=2,
        dropout=0.1,
        time_encoding="relative",
        time_dim=16,
        time_periods=(1.0, 7.0, 28.0),
        history_days=21,
    ):
        super().__init__()
        if time_encoding != "relative":
            raise ValueError("Unknown time encoding")
        if (
            history_days <= 0
            or time_dim < 1
            or not time_periods
            or any(not np.isfinite(p) or p <= 0 for p in time_periods)
        ):
            raise ValueError(
                "Time dimensions/history/periods must be positive and finite"
            )
        self.time_encoding = time_encoding
        self.history_days = history_days
        self.layers = layers
        self.dropout = dropout
        self.encoders = nn.ModuleList(nn.Linear(FIELDS, hidden) for _ in TYPES)
        self.self_linears = nn.ModuleList(
            nn.ModuleList(nn.Linear(hidden, hidden) for _ in TYPES)
            for _ in range(layers)
        )
        self.messages = nn.ModuleList(
            nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in RELATIONS)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleList(nn.LayerNorm(hidden) for _ in TYPES) for _ in range(layers)
        )
        self.head = nn.Linear(hidden, 6)
        self.register_buffer(
            "time_periods", torch.tensor(time_periods, dtype=torch.float32)
        )
        self.time_encoders = nn.ModuleDict(
            {
                str(i): nn.Sequential(
                    nn.Linear(1 + 2 * len(time_periods), time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, hidden),
                )
                for i in range(2, len(TYPES))
            }
        )

    def time_features(self, age):
        """Age in fractional days, with untruncated parent age and fixed cycles."""
        phase = 2 * torch.pi * age[:, None] / self.time_periods[None, :]
        scaled_log = torch.log1p(age) / np.log1p(self.history_days)
        return torch.cat([scaled_log[:, None], phase.sin(), phase.cos()], dim=-1)

    def forward(self, x, edges, roots, ages=None):
        if self.time_encoding == "relative" and ages is None:
            raise ValueError(
                "Relative encoding requires v2 cache; rerun run_adaptive.py prepare"
            )
        encoded = []
        for i, (enc, value) in enumerate(zip(self.encoders, x)):
            h = enc(value)
            if self.time_encoding == "relative" and i >= 2:
                h = h + 0.1 * self.time_encoders[str(i)](self.time_features(ages[i]))
            encoded.append(F.relu(h))
        x = encoded
        for layer in range(self.layers):
            output = [
                linear(value) for linear, value in zip(self.self_linears[layer], x)
            ]
            for r, (src, dst) in enumerate(RELATIONS):
                a, b = edges[r]
                # Aggregate before projection to avoid one dense transform/edge.
                pooled = x[dst].new_zeros(x[dst].shape)
                pooled.index_add_(0, b, x[src][a])
                degree = torch.bincount(b, minlength=len(pooled)).clamp_min(1)
                output[dst] = output[dst] + self.messages[layer][r](
                    pooled / degree[:, None]
                )
            x = [
                F.dropout(F.relu(norm(value)), self.dropout, self.training)
                for norm, value in zip(self.norms[layer], output)
            ]
        return self.head(x[0][roots])
