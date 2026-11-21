"""Official six-label F1, calibration and paired uncertainty estimates."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import TARGETS


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    tp = (truth & prediction).sum(axis=0)
    fp = (~truth & prediction).sum(axis=0)
    fn = (truth & ~prediction).sum(axis=0)
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.ones(6, float), where=den > 0)
    micro = 2 * tp.sum() / den.sum() if den.sum() else 1.0
    return {
        "macro_f1": float(f1.mean()),
        "micro_f1": float(micro),
        "score": float((f1.mean() + micro) / 2),
        "f1": f1.tolist(),
        "tp": tp.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
        "support": truth.sum(axis=0).tolist(),
    }


def calibrate(truth, probabilities):
    """Coordinate search of the official score on calibration weeks only."""
    thresholds = np.full(6, 0.5)
    best = metrics(truth, probabilities >= thresholds)["score"]
    for _ in range(3):
        for label in range(6):
            for value in np.linspace(0.05, 0.8, 31):
                candidate = thresholds.copy()
                candidate[label] = value
                score = metrics(truth, probabilities >= candidate)["score"]
                if score > best + 1e-12:
                    thresholds, best = candidate, score
    return thresholds


def save_prediction(directory, split, data, probabilities, thresholds):
    prediction = probabilities >= thresholds
    score = metrics(data["y"], prediction)
    score["thresholds"] = np.asarray(thresholds).tolist()
    score["average_precision"] = [
        float(average_precision_score(data["y"][:, j], probabilities[:, j]))
        if data["y"][:, j].sum()
        else None
        for j in range(6)
    ]
    score["at_0_5"] = metrics(data["y"], probabilities >= 0.5)
    score["weeks"] = {}
    for week in np.unique(data["weeks"]):
        take = data["weeks"] == week
        score["weeks"][week] = metrics(data["y"][take], prediction[take])
    output = pd.DataFrame({"cutoff_utc": data["weeks"], "UserID": data["ids"]})
    for j, name in enumerate(TARGETS):
        output[f"truth_{name}"] = data["y"][:, j]
        output[f"p_{name}"] = probabilities[:, j]
        output[name] = prediction[:, j].astype(int)
    output.to_csv(directory / f"{split}_predictions.csv.gz", index=False)
    (directory / f"{split}_metrics.json").write_text(json.dumps(score, indent=2))
    return score
