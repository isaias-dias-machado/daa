# Frozen cross-validation folds

`splits/folds.npz` is generated from the training target and committed so every
run in the group uses exactly the same folds.

- **Scheme:** repeated stratified 5-fold, 3 repeats (15 fits per model).
- **Seed:** 42.
- **File contents:** an `int8` assignment matrix of shape `(3, 10962)`, the
  target fingerprint, and the split parameters.

Load it through `daa.splits.get_folds(y)`. The function raises a `ValueError`
if the target vector no longer matches the stored fingerprint.

Regenerate deliberately (this invalidates every previous score):

```bash
python -m daa.splits --force
```

Do not edit the file by hand.
