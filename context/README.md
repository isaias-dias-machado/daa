# Context pack — read this first

This directory is a map, not a library. Open only the file that answers your question.

## Router

| Question | File |
|---|---|
| What is required, when is it due, how is it assessed? | `assignment/enunciado.md` |
| Give me the requirement checklist | `assignment/deliverables.md` |
| Competition rules, target coding, submission format? | `competition/kaggle.md` |
| Dataset shapes, class counts, ID rules? | `competition/data-dictionary.md` |
| What did we already find in the data? | `../reports/01_feature_selection_review.md` |
| Which slide deck covers this method, and on which pages? | `notes/slides-index.md` |
| Corrected study notes | `notes/ml-reference.md` |
| What changed from the original Portuguese notes? | `notes/notas-audit.md` |
| What is still unresolved? | `../TODO.md` |

## Authority is per fact

| Fact | Owner | Fallback |
|---|---|---|
| Requirements, deadlines, assessment | `assignment/enunciado.md` | `assignment/deliverables.md` (checklist) |
| Competition rules, submission, target coding | `competition/kaggle.md` | the Kaggle page |
| Measured dataset facts | `data/` + `competition/data-dictionary.md` | — |
| Analysis of current data | `../reports/` | — |
| Study material | `notes/` | the slides |

A derived document is never the source. If two documents disagree, the owner wins and
the conflict is logged here.

## Conflict log

| Conflict | Status | Owner |
|---|---|---|
| Submission header is `RowId,AGE_RANGE`, CRLF, 3,655 rows, `RowId` 0…3654 | resolved | `competition/kaggle.md` |
| `AGE_RANGE` band 0 is disputed (official `15–17` vs data-consistent `0–14`) | **open** | `../TODO.md` |
| Test file is `dataset_test_X.csv`, not `dataset_test.csv` | resolved | `competition/data-dictionary.md` |
| 475 = 237 + 236 + 2 (sum, not union) | resolved | `competition/data-dictionary.md` |
| Deadlines are absolute, not “2 months to go” | resolved | `assignment/deliverables.md` |
| There are 2 tasks, a 20-page report, a checkpoint, presentations, peer assessment | resolved | `assignment/enunciado.md` |
| `reports/01` outputs are from `data_v1/`, not regenerated | nuance | `../reports/01_feature_selection_review.md` |

## Rules

- Do not bulk-load this directory. Read this file, then open one document at a time.
- PDFs under `slides/` are binary. Text extraction loses diagrams. Use
  `notes/slides-index.md`, then extract only the pages you need:
  `pdftotext -layout -f N -l M "slides/<file>" -`.
- `notes/` is study material, not requirements.
- `data/` is local-only. If it is missing, stop and ask; do not synthesize data.
- The age-band coding is open; see `../TODO.md`.

## Known stale

- `reports/01`'s stored outputs come from `data_v1/` (238 columns, includes `WGT`).
  Current `data/` has 237 columns and no `WGT`; the outputs were not regenerated.
- The Kaggle page's “2 months to go” is not copied anywhere.
