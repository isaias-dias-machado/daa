# Project context

Read `context/README.md` first. It maps each question to the file that answers it and
states the per-fact authority order.

- Do not bulk-load `context/`. Read the index, then open only what the task needs.
- The default-loaded files are this file and `context/README.md` (target ≤ 5 KB
  total). Everything else is on demand.
- PDFs under `context/slides/` are binary; text extraction loses diagrams. Use
  `context/notes/slides-index.md` to pick the deck and page range, then extract only
  those pages: `pdftotext -layout -f N -l M "context/slides/<file>" -`.
- `context/notes/` is study material, not requirements. For requirements use
  `context/assignment/enunciado.md`; for competition rules use
  `context/competition/kaggle.md`; for data facts use `data/` and
  `context/competition/data-dictionary.md`.
- `data/` is local-only and not version-controlled. If it is missing, stop and ask;
  do not synthesize data. In use: `dataset_train.csv` (10,962 × 237),
  `dataset_test_X.csv` (3,655 × 236), `example_submission_10.csv` (format template;
  every value is a placeholder). `data_v1/` does not exist.
- The `AGE_RANGE` band coding is open; see `TODO.md`. Do not state a band mapping as
  fact.
