# Submission checklist (revised Assignment 2)

## Public GitHub code

- [ ] Publish all retained source code and README before Nov 21, 2026 23:55 Taiwan time.
- [x] README explicitly acknowledges GH Archive as Data Source.
- [x] Instructor preprocessing/evaluator, schema and cohort are retained.
- [x] Code covers table reading, feature construction, train, calibration, ablation and prediction.
- [ ] Confirm the instructor's latest test_input.csv; the retained copy has 4,465 users.
- [ ] Install from documented requirements and reproduce the selected model/output.
- [ ] Keep trained checkpoint, run.json, environment.json and CSV metadata for reproduction.
- [ ] Record the actual public repository URL for the presentation slides.

No public repository URL or publication is fabricated by this refactor.

## E3 prediction

- [ ] Generate `output_314561001.csv` (replace student ID if needed).
- [ ] Predict Nov 22 00:00–Nov 29 00:00, Taiwan time.
- [ ] Use only officially preprocessed history available before prediction cutoff.
- [ ] Run official evaluate.py with the exact latest test user list.
- [ ] Submit exactly one row per requested user, correct header, six integer 0/1 labels.
- [ ] Submit the CSV to E3.

The revised grading gives 1% for the README Data Source acknowledgement; the
F1 Score is reference information. The TAs still check output reproducibility.

## Next step after real runs: slides and video

- [ ] English slides using the instructor's template, no more than 16 pages.
- [ ] File: `Asg2_314561001.pdf`.
- [ ] Video: 16–20 minutes, spoken Mandarin or English, recorded by the student
      using Webex/Google Meet; file `Asg2_314561001.mp4`.
- [ ] Include actual public GitHub URL on slides.
- [ ] Q1: full schema graph with 8 tables, every FK, PK/FK labels; actual data time period (1%).
- [ ] Q2: cross-table relationship modeling and supporting experiments (3%).
- [ ] Q3: temporal modeling and supporting experiments (5%).
- [ ] Q4: framework figure and full pipeline (5%).
- [ ] Content used for grading appears in both slides and spoken video.
- [ ] Report only implemented mechanisms and measured results.

Framework slides, final report/video and E3/GitHub publishing are pending the
user's real run results and chosen model, as requested.
