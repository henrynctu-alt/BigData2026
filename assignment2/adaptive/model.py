"""Fuse typed relational messages and multi-scale causal convolutions."""

import torch
from torch import nn
from torch.nn import functional as F

from relational.routes import ROUTES

from .features import DAILY_FIELDS, EDGE_FIELDS, LABELS

VARIANTS = ("base", "graph", "temporal", "additive", "adaptive")


def to_device(batch, device):
    """Move every tensor in one minibatch, including masks and labels."""
    return {key: value.to(device) for key, value in batch.items()}


def masked_pool(values, scores, valid):
    """Pool by attention. Return zeros for empty sets."""
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weight = scores.softmax(dim=-1) * valid
    return (values * weight.unsqueeze(-1)).sum(dim=-2)


class GraphEncoder(nn.Module):
    """Neighbor → typed route → destination user message passing."""

    def __init__(self, hidden):
        super().__init__()
        self.register_buffer("roles", torch.arange(len(ROUTES)))
        self.role = nn.Embedding(len(ROUTES), hidden)
        self.message = nn.Linear(len(EDGE_FIELDS), hidden)
        self.attention = nn.Linear(hidden, 1)
        self.reliability = nn.Linear(len(EDGE_FIELDS), 1)
        self.route_attention = nn.Linear(hidden, 1)

    def forward(self, edges, mask):
        role = self.role(self.roles)[None, :, None, :]
        message = F.gelu(self.message(edges) + role)
        scores = self.attention(message).squeeze(-1)
        reliability = torch.sigmoid(self.reliability(edges))
        route = masked_pool(message * reliability, scores, mask)
        return masked_pool(
            route, self.route_attention(route).squeeze(-1), mask.any(dim=2)
        )


class TemporalEncoder(nn.Module):
    """Encode daily counts and attend to multiple time scales."""

    def __init__(self, hidden, days):
        super().__init__()
        self.windows = sorted({min(days, k) for k in (1, 3, 7, 14, 21, days)})
        self.input = nn.Linear(len(DAILY_FIELDS), hidden)
        self.convs = nn.ModuleList(
            nn.Conv1d(hidden, hidden, 3, dilation=d) for d in (1, 2, 4, 8)
        )
        self.day_score = nn.Linear(hidden, 1)
        self.scale = nn.Linear(hidden + LABELS + 3, hidden)
        self.scale_id = nn.Embedding(len(self.windows), hidden)
        self.scale_score = nn.Linear(hidden, 1)

    def forward(self, daily):
        data = self.input(daily).transpose(1, 2)
        for layer in self.convs:
            left = (layer.kernel_size[0] - 1) * layer.dilation[0]
            data = data + F.gelu(layer(F.pad(data, (left, 0))))
        states = data.transpose(1, 2)
        counts = torch.expm1(daily[..., :LABELS])
        scales = []
        for i, span in enumerate(self.windows):
            window = states[:, -span:]
            att = self.day_score(window).softmax(dim=1)
            profile = counts[:, -span:].sum(dim=1)
            volume = counts[:, -span:].sum(dim=-1)
            share = volume / volume.sum(dim=1, keepdim=True).clamp_min(1)
            entropy = -(share * share.clamp_min(1e-8).log()).sum(dim=1)
            features = torch.cat(
                [
                    (window * att).sum(dim=1),
                    torch.log1p(profile / span),
                    (volume > 0).float().mean(dim=1, keepdim=True),
                    share.amax(dim=1, keepdim=True),
                    entropy[:, None],
                ],
                dim=1,
            )
            scales.append(F.gelu(self.scale(features) + self.scale_id.weight[i]))
        scales = torch.stack(scales, dim=1)
        return (scales * self.scale_score(scales).softmax(dim=1)).sum(dim=1)


class DualBranchNet(nn.Module):
    """Six base logits plus enabled branch and interaction residuals."""

    def __init__(self, variant, days, hidden=32, branch_dropout=0.1):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown model {variant}")
        self.variant = variant
        self.branch_dropout = branch_dropout
        self.register_buffer("mean", torch.zeros(LABELS))
        self.register_buffer("std", torch.ones(LABELS))
        self.base = nn.Sequential(nn.Linear(LABELS, hidden), nn.GELU())
        self.prior = nn.Linear(hidden, LABELS)
        self.heads = nn.ModuleDict()
        if variant in ("graph", "additive", "adaptive"):
            self.graph = GraphEncoder(hidden)
            self.heads["graph"] = nn.Linear(hidden, LABELS)
        if variant in ("temporal", "additive", "adaptive"):
            self.temporal = TemporalEncoder(hidden, days)
            self.heads["temporal"] = nn.Linear(hidden, LABELS)
        if variant == "adaptive":
            self.heads["interaction"] = nn.Linear(hidden, LABELS)
            self.gates = nn.Linear(3 * hidden, 3 * LABELS)
            nn.init.constant_(self.gates.bias, -1.0)
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, base, daily, edge, mask):
        anchor = self.base((base - self.mean) / self.std)
        logits = self.prior(anchor)
        states = {}
        enabled = {}
        if "graph" in self.heads:
            states["graph"] = self.graph(edge, mask)
            enabled["graph"] = mask.any(dim=(1, 2))[:, None]
        if "temporal" in self.heads:
            states["temporal"] = self.temporal(daily)
            enabled["temporal"] = base.sum(dim=1, keepdim=True) > 0
        if self.variant == "adaptive":
            for name in ("graph", "temporal"):
                if self.training and self.branch_dropout:
                    keep = torch.rand_like(base[:, :1]) >= self.branch_dropout
                    enabled[name] = enabled[name] & keep
                states[name] = states[name] * enabled[name]
            g, t = states["graph"], states["temporal"]
            states["interaction"] = g * t
            enabled["interaction"] = enabled["graph"] & enabled["temporal"]
            gate = torch.sigmoid(self.gates(torch.cat([anchor, g, t], dim=1)))
            weights = gate.reshape(-1, 3, LABELS)
        for i, (name, state) in enumerate(states.items()):
            delta = self.heads[name](state) * enabled[name]
            if self.variant == "adaptive":
                delta = delta * weights[:, i]
            logits = logits + delta
        return logits
