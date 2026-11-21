# Assignment 2 — Predicting GitHub User Activity

A compact source repository for preprocessing, EDA, seven-model ablation,
and official prediction. All model stages run through `run_adaptive.py`.

## Data Source

**GH Archive: https://www.gharchive.org/**. Use only the eight tables generated
by the instructor's `official/scripts/preprocess.py`. No GitHub API, extra event
types or external user/repository features are used.

## Retained models

| CLI name | Model |
|---|---|
| copy7 | Repeat the six activities observed in the previous 7 days; no training |
| base | MLP on the six user event counts |
| graph | Base plus typed-route relational encoder |
| temporal | Base plus causal daily TCN |
| additive | Base plus Graph and Temporal residuals |
| adaptive | Gated Graph/Temporal residuals and interaction |
| gnn_time | Table-row heterogeneous GraphSAGE with relative time encoding |

`addaptive` aliases `adaptive`; `gnn_relative` aliases `gnn_time`.
GNN time encoding uses log-age and sin/cos with periods 1, 7 and 28 days,
projected by node type before GraphSAGE. There are eight node types and 30
directed author/repository/parent relations including reverse edges. This is
an FK subset, not the complete schema graph. The complete schema is generated
separately in `docs/assets/schema.svg`.

## Install

Python 3.11+ is recommended (verified with 3.12). Install PyTorch >=2.6 with the
CPU/CUDA build appropriate for your machine using https://pytorch.org/get-started/locally/,
then:

```bash
python -m pip install -r requirements.txt
# For tests and lint:
python -m pip install -r requirements-dev.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Graphviz `dot` is needed only for schema rendering (e.g. Ubuntu: install the
`graphviz` system package). GNN does not require PyG or pyg-lib.

## Data preparation

Download GH Archive hourly files or use your existing local archives. Hours
in the downloader are UTC, with an exclusive end:

```bash
python download.py --start 2026-11-14T16 --end 2026-11-21T16 \
  --out raw/recent --manifest data/recent_download_manifest.json
python official/scripts/preprocess.py --raw raw/recent --out data/recent
```

The preprocessor writes `data/recent/db/{Users,Repositories,PullRequests,Issues,
IssueComments,PRReviews,PRReviewComments,Pushes}.parquet`. Repeat for historical
sources if necessary. Existing `train_db.zip`, `valid_db.zip`, `test_db.zip`
from the current preprocessing version may be placed in `data_inputs/`.
Raw data, prepared tables, caches and previous experimental outputs are not
part of this source archive or the public Git repository.

Each `--db` argument is either a directory directly containing the eight
Parquet tables, or a ZIP containing exactly one of each table. Source order is
chronological. Primary/foreign keys are resolved inside each source before
combining by stable anonymized GitHub IDs. Event duplicates are removed.

## Run the complete historical comparison

With the three locally preprocessed archives:

```bash
bash scripts/run_historical.sh runs/full cuda:0 \
  data_inputs/train_db.zip data_inputs/valid_db.zip data_inputs/test_db.zip
```

This runs EDA, schema generation, shared snapshot preparation, training and
validation/test evaluation of all seven models. Start with a new output path.
For individual stages:

```bash
python run_adaptive.py prepare \
  --db data_inputs/train_db.zip data_inputs/valid_db.zip data_inputs/test_db.zip \
  --plan configs/historical.json --data runs/full/data

python run_adaptive.py train \
  --data runs/full/data --out runs/full/models --device cuda:0

python run_adaptive.py evaluate \
  --data runs/full/data --models runs/full/models \
  --split evaluation --out runs/full/validation --device cuda:0

python run_adaptive.py evaluate \
  --data runs/full/data --models runs/full/models \
  --split test --out runs/full/test --device cuda:0
```

The default train set is `copy7 base graph temporal additive adaptive gnn_time`.
To train a subset, pass e.g. `--models copy7 base gnn_time`.
Defaults: epochs 16, patience 4, lr .002, batch 256, seed 2026, weighted BCE.
Use `--loss bce` for unweighted BCE. Adaptive hidden=32; GNN hidden=64,
layers=2, fanout=8, dropout=.1, time-dim=16. All settings, thresholds and
parameter counts are recorded in each model's `run.json`.

## Time plan and leakage controls

`configs/historical.json` preserves the uploaded configuration: **H=7 days**,
train first cutoff Jan 1, stride 7 days, last bound Jul 19, 2026. Because the
first cutoff is a Thursday, actual training cutoffs end Jul 16 (29 cutoffs).
Stop: Aug 2/9; calibration: Aug 16/23; validation: Sep 6/13/20; historical test:
Oct 4/11/18/25 and Nov 1/8. These dates are at 00:00 Taiwan time (UTC+8).
The observation start is Dec 1, 2025 UTC; observation end is Nov 15, 2026
00:00 Taiwan time. Inspect/edit the plan before generating new caches.

At cutoff t, model inputs use event time in [t-H,t), observation time <t.
Parent issue/PR endpoints can predate H if already known before t. GNN retains
their true age for the time encoder. Targets are six independent labels over
[t,t+7 days), observed before label-week end. Late close/merge fields are
excluded. Missing source data must not be interpreted as true inactivity.

Choose epoch by stop BCE, fit six thresholds on calibration only, select
architecture/hyperparameters using validation, then report locked test results.
Changing H or split dates requires prepare and train again in new folders.
New unified caches/checkpoints are not interchangeable with old model runs.

`--coverage hourly.csv` optionally adds known/observed coverage channels to the
Temporal branch; use the same policy for prepare and predict. GNN has no direct
coverage channel. CSV columns: hour_utc,available_at,observed (0/1).

## Read the comparison

Each evaluate output contains:

- `summary.csv`: pooled Macro-F1, Micro-F1, Score, sample count and parameters.
- `weekly.csv`: pooled and per-cutoff scores for every model.
- `per_label.csv`: per-label AP, F1, support, TP/FP/FN, pooled and per cutoff.
- `<model>/<split>_predictions.csv.gz`: user-week truth/probability/decision.
- `<model>/<split>_metrics.json`: thresholds, AP, confusion counts and score at .5.

Score = (Macro-F1 + Micro-F1)/2, including the official empty-class convention.
For cross-run comparison, `compare_runs.py --runs RUN1 RUN2 --split test --out
comparison.csv` refuses mismatched user/cutoff/truth keys. Evaluate defaults to
the batch size saved in the checkpoint because GNN sampling depends on batch
grouping; keep it unchanged for reproducibility.

## Evaluate a new historical week with frozen checkpoints

```bash
python run_adaptive.py evaluate-week \
  --data runs/full/data --models runs/full/models \
  --db data_inputs/train_db.zip data_inputs/valid_db.zip data_inputs/test_db.zip \
  --cutoff 2026-09-27T00:00:00+08:00 \
  --out runs/full/sep27 --device cuda:0
```

All seven models use exactly the same users and target week, without changing
thresholds. Sources must match those of prepare; the target week must end
within observation_end. This is retrospective evaluation, separate from the
future submission week.

## Produce the official submission

Official target: **Nov 22, 2026 00:00 to Nov 29, 2026 00:00 Taiwan time**.
Choose the checkpoint from validation evidence. Example using `adaptive`:

```bash
python run_adaptive.py predict \
  --checkpoint runs/full/models/adaptive/model.pt \
  --db data_inputs/test_db.zip data/recent/db \
  --users official/submission/test_input.csv \
  --cutoff 2026-11-22T00:00:00+08:00 \
  --out outputs/output_314561001.csv --device cuda:0

python official/scripts/evaluate.py \
  --pred outputs/output_314561001.csv \
  --input official/submission/test_input.csv
```

For GNN use `runs/full/models/gnn_time/model.pt`; the command is otherwise the
same. Copy7 also supports this predict command using its saved checkpoint.
The script writes exactly `UserID,open_pr,open_issue,issue_comment,pr_review,
pr_review_comment,push`, in input user order with integer binary predictions.
It also writes a `.metadata.json` sidecar with cutoff, checkpoint/source hashes,
thresholds and inference batch size. Only the CSV is submitted as predictions.

The supplied official list has 4,465 users. Verify the latest instructor test
list before final submission. History archives in the original configuration
end Nov 15; add officially preprocessed recent data through the prediction
cutoff to supply the intended H-day history. No future target-week data should
be used. Source table availability alone cannot establish archive completeness.
No final output CSV is fabricated in this refactor; generate it from your
chosen trained checkpoint and available history.

## EDA and schema

```bash
python run_eda.py \
  --db data_inputs/train_db.zip data_inputs/valid_db.zip data_inputs/test_db.zip \
  --plan configs/historical.json --out reports/eda
python schema_graph.py --out reports/schema
```

EDA includes daily/hourly counts, weekly prevalence/Copy7, label correlation,
activity recurrence, user-repository connectivity and observation delays.
Target-distribution EDA ends with validation; test tables receive schema/quality
audits. Zero activity is not a measure of archive completeness.
The schema tool derives all 8 tables, PKs and every FK from official TABLES,
including links not used by the GNN.

## Quick pilot and tests

```bash
python run_adaptive.py prepare --db data_inputs/train_db.zip \
  --plan configs/pilot.json --sample-users 96 --data runs/pilot/data
python run_adaptive.py train --data runs/pilot/data --out runs/pilot/models \
  --epochs 3 --hidden 24 --gnn-hidden 24 --gnn-fanout 4 --batch-size 32 --device cpu
python run_adaptive.py evaluate --data runs/pilot/data --models runs/pilot/models \
  --split evaluation --out runs/pilot/validation --device cpu
python -m pytest -q
ruff check .
```

The pilot is a flow check, not a full historical performance claim. See
`docs/VALIDATION.md` for the checks actually completed on this refactor.

## Return results for the report/video step

```bash
python scripts/pack_results.py --run runs/full --out outputs/results_for_report.zip \
  --predictions outputs/output_314561001.csv outputs/output_314561001.csv.metadata.json
```

This includes configurations, metrics, predictions, EDA and logs, excluding
large cutoff caches and model weights. Add `--include-checkpoints` if needed.
Follow `docs/SUBMISSION_CHECKLIST.md` before publishing code or submitting to E3.
No GitHub publication or E3 upload is performed by these commands.
