# Refactor map

| Original component | Retained location / decision |
|---|---|
| Instructor preprocessing/evaluator, cohort, test input, schema | `official/`, original bytes preserved |
| `adaptive/model.py` encoders, fusion and five learned variants | Same path, original computation retained |
| `adaptive/features.py` | Same features, adds unified cache schema version |
| `relational/data.py`, `evaluation.py` | Canonical key resolution, metrics and calibration retained |
| `relational/features.py` | Only `ROUTES` is needed by the retained architecture; moved to `relational/routes.py` |
| `relational_gnn.py` | `adaptive/gnn.py`, retains relative time encoding only |
| Separate `run_gnn.py` | Removed; all stages dispatch through `run_adaptive.py` |
| `evaluate_adaptive_week*.py` | Unified `run_adaptive.py evaluate-week`, includes GNN |
| `copy7` | Explicit zero-parameter checkpoint and inference in unified workflow |
| EDA | `run_eda.py`, same daily/hourly drift, recurrence, availability and relationship analyses; plan-based dates and ZIP support |
| Schema diagram | `schema_graph.py`, reads all PK/FK definitions from official code |
| Cross-run comparison | Renamed to `compare_runs.py`; same-run evaluation now writes one comparison for all seven models |
| Historical config | Preserved from uploaded hw2.zip: H=7; train first Jan 1, stride 7, upper bound Jul 19 |
| Other experiments | Graph28, OR14, LightGBM, logistic/tabular/sliding pipelines, stale reports/checkpoints/caches removed |

New caches and checkpoints must be generated. Old caches/checkpoints belong to
older manifest schemas; their results are not copied into the submission as
new experimental results. `adaptive` is the canonical name; CLI accepts the
user spelling `addaptive` as an alias. `gnn_time` is the retained GNN name;
`gnn_relative` is an input alias.

The independent old GNN and adaptive engines used different metadata schemas.
The unified prepare now builds adaptive and row-graph tensors at the same cutoff,
asserts identical user/label/Copy7 rows, then stores them in one NPZ file.

Model hyperparameters are separate: --hidden/--branch-dropout for the original
five learned variants, --gnn-hidden/--gnn-layers/--gnn-fanout/--gnn-dropout and
--time-dim/--time-periods for GNN. All share epochs/patience/lr/loss/seed and splits.
Different capacities are recorded; comparison is of complete pipelines, not a
parameter-matched isolation of the graph encoder.
