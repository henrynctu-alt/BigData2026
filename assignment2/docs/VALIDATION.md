# Verification of this refactor

## Completed checks

- 11 automated tests passed; 1 CUDA-only test skipped on this CPU environment.
- Ruff passed for all retained project code (official upstream code is excluded).
- Instructor files preserved byte-for-byte; hashes in verification/official_sha256.json.
- Synthetic raw GitHub events passed through the original instructor table builder,
  anonymization and Parquet writer; unified prepare/train/evaluate/predict exercised
  every one of the seven retained model names.
- Each model prediction matched its evaluation export on the same historical week.
- Every model's F1 Score matched the original instructor scorer on that fixture.
- Unified evaluate-week matched cached evaluation; mismatched data IDs were rejected.
- Future and late events are excluded; old parent time features and time-encoder
  gradient flow, and isolated user/empty graph behavior were tested.
- Actual eight-table train_db.zip used for a 96-user pilot: prepare, EDA, all seven
  models trained for 3 epochs, then pooled/weekly/per-label evaluation exported.
- GNN prediction on all 4,465 official users at a historical cutoff passed the
  original official evaluate.py format check. This is not the final November CSV.
- All eight schema nodes and all official FK edges generated to SVG/PDF/DOT;
  rendered figure inspected.

## Actual pilot configuration

H=21, sampled users=96, seed=2026, batch=32, hidden=24, GNN hidden=24,
GNN layers=2, fanout=4, epochs=3, CPU. Training cutoffs Dec 28/Jan 4;
stop Jan 11; calibration Jan 18; evaluation Jan 25, 2026 (00:00 UTC+8).
Scores in verification/pilot_summary.csv verify the running pipeline. They
are not a full historical evaluation, and do not replace the user's new run.
The uploaded full historical config remains H=7.

Environment used: Python 3.12.14, PyTorch 2.15.0+cpu, NumPy 2.3.5,
pandas 2.2.3, PyArrow 26.0.1, scikit-learn 1.8.0, matplotlib 3.10.8.
GPU training and complete 29-cutoff training were not run in this environment.

The final code archive excludes old scores, caches and pretrained weights from
the uploaded experimental repository. The small verification evidence above
is newly produced by the refactored pipeline.
