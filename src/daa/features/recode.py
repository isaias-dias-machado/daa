"""Recoding: turn the raw survey text into usable numbers.

The survey export stores many numbers as text. ``BM2`` (body weight) is a
string column with 102 levels, and ``'70.0'`` is one of those levels. A model
that one-hot encodes it learns "the level 70.0", not "70 kilograms". This
module fixes the representation before any model sees the frame.

Two ideas drive every rule here:

1. Some columns are quantities in disguise. They are parsed to ``float``.
2. Some cells are not answers. ``'Sem resposta'`` means "no answer" and
   ``'Não aplicável'`` means "this question did not apply to this person".
   These are *routing* facts. The survey skips questions, and who was skipped
   is often related to age. So the fact is useful and must not be erased: each
   parsed column gets a ``<COL>_FLAG`` companion column that keeps it.

Design principle (why this runs at load time, outside the CV folds):

    A transform leaks when it **estimates** something from rows it must not
    see (a median, a category frequency). Recoding is a fixed rule-book: the
    same input value always produces the same output value, no matter which
    rows are in the frame. Nothing is estimated, so nothing can leak.
    Therefore recoding may run at load time, outside the CV folds. Anything
    that **learns from data** (imputation, rare-level grouping, scaling,
    target encoding) stays in ``preprocess.py`` inside the pipeline, where it
    is refitted on each fold.

What is parsed in v1 (29 columns, so 29 flag columns):

- ``HHMM_COLUMNS``: fixed-width ``'HHMM'`` text read as a **duration**.
- ``QUANTITY_COLUMNS``: plain numbers stored as text.
- ``BUCKET_COLUMNS``: numbers plus a ``'>=N'`` / ``'N ou mais'`` bucket. The
  bucket becomes its floor ``N``. This is an approximation, not a true value.

What is left nominal on purpose: ``PA1`` (``YYYYMM`` plus text buckets),
``IN24`` (years), ``IN21`` and ``IN62`` (multi-select categories). Their
meaning is not confirmed yet, so parsing them would be a guess.

Units and meaning are NOT confirmed by a codebook. The data shows the shape;
the survey documentation must confirm the meaning. Every parsed column carries
a one-line comment saying what the data shows and what is still open. Do not
state any of those interpretations as fact in the report.

How this may change when we optimise for the best score:

1. ``PE7`` / ``PE9``: the duration reading is confirmed by the data (values
   above 23:59 exist, e.g. ``'3000'`` = 30h00), so no cyclical sin/cos
   encoding is needed and the linear minutes are final. Still open: which
   activity the duration measures. Verify against the survey documentation
   before interpreting results in the report.
2. If the 29 flag columns bloat linear models, collapse them to a single
   binary "answered" flag. Keep the full flags for trees.
3. Parse the left-nominal columns once their meaning is confirmed: ``PA1``
   (``YYYYMM`` plus buckets, could become "months since"), ``IN24`` (years,
   could become "years since"), ``IN21`` / ``IN62`` (split into binary
   indicators). ``HO12_COD`` / ``HO34_COD`` are already parsed as buckets in
   v1; any rare-code grouping is a learned transform and belongs in
   ``preprocess.py``, not here.
4. Ordinal promotion of rating scales (``HS1``-style) lives in
   ``explicit_spec`` in ``data.py``, never in this module. The long
   ``'Não aplicável (...)'`` variant in ``HA1A``-``HA1G`` must be handled
   before those columns are promoted.
5. Anything that learns from the data (rare-level grouping, target encoding)
   must never be added here. It belongs in ``preprocess.py`` inside the CV
   pipeline, or it leaks.
6. Derived features (for example BMI from ``BM1`` and ``BM2``) are feature
   engineering, a future separate module, not recoding.
"""

from __future__ import annotations

import re

import pandas as pd

# Answers that are not numbers. The flag column keeps them as levels.
# Known variants outside the v1 parsed columns (not added yet, because they
# never reach the guard today): 'NR' in IN19, PA5, SK5; the long
# 'Não aplicável (nunca tentou ou não precisa de o fazer)' in HA1A-HA1G.
# Add a variant here only when its column is parsed, with an evidence comment.
# Open question for the codebook: which of these are routing (the question was
# skipped) and which are non-response (the person refused)? The flag meaning
# changes if we get it wrong.
SENTINELS = ("Sem resposta", "Não aplicável")

# Levels of a ``<COL>_FLAG`` column. "missing" is a real gap in the file, not
# an answer. It is unused in v1 because only IN16 has real NaN and IN16 is not
# parsed; the level stays as a future-proof rule.
FLAG_ANSWER = "answer"
FLAG_MISSING = "missing"

# Fixed-width 'HHMM' text, read as a duration: 60*HH + MM minutes.
# Evidence: every non-sentinel value has exactly 4 digits, in train and test.
# Values above 23:59 exist ('2400', '2425', '2500', '3000'), which is
# impossible on a clock, so this is a duration (hours and minutes), not a
# time of day. Half-hour codes exist ('0130') and raw-minute codes ('0060')
# never appear, which supports the same reading.
# PE7 max is 1,800 minutes (30h), PE9 max is 1,440.
# Open question: which activity does the duration measure, and is it a weekly
# total? The parse is safe; the report interpretation is not.
HHMM_COLUMNS = ("PE7", "PE9")

# Plain numbers stored as text. The sentinels above become NaN.
# Open questions: units and meaning; is the value a count, an amount, or a
# code? A code treated as a quantity imposes a false order.
QUANTITY_COLUMNS = (
    "BM1",  # height-like values; open: units (cm?) and self-reported?
    "BM2",  # weight-like values; open: units (kg?) and self-reported?
    "IN55",  # many levels; open: count, amount, or code?
    "IN56",  # many levels; open: count, amount, or code?
    "IN57",  # many levels; open: count, amount, or code?
    "IN45",  # many levels; open: count, amount, or code?
    "IN4",  # many levels; open: count, amount, or code?
    "IN3",  # many levels; open: count, amount, or code?
    "IN59",  # many levels; open: count, amount, or code?
    "IN2",  # many levels; open: count, amount, or code?
    "IN47",  # many levels; open: count, amount, or code?
    "SK2",  # integer levels; open: rating scale (ordinal) or count?
    "AM3",  # integer levels; open: rating scale (ordinal) or count?
    "AM5",  # integer levels; open: rating scale (ordinal) or count?
    "DH2",  # integer levels; open: rating scale (ordinal) or count?
    "DH4",  # integer levels; open: rating scale (ordinal) or count?
    "PE2",  # integer levels; open: rating scale (ordinal) or count?
    "PE4",  # integer levels; open: rating scale (ordinal) or count?
    "PE6",  # integer levels; open: rating scale (ordinal) or count?
    "PE8",  # integer levels; open: rating scale (ordinal) or count?
    "IN29",  # only 4 levels; open: rating scale or count? ordinal vs numeric
)

# Numbers plus a top bucket. The bucket is open-ended, so the exact value is
# unknown and we use the only number we can defend: the floor N.
# Evidence: '>=7' 40 rows, '>=3' 88, '53 ou mais' 15, '33 ou mais' 5,
# '>= 25' 93, '>= 60' 11. The whitespace after '>=' is allowed.
# The '_COD' suffix does not mean "leave nominal": HHNBPERS_COD is parsed too.
# HO34_COD has a test-only '33.0'; parsing makes it harmless.
BUCKET_COLUMNS = (
    "HHNBPERS_COD",  # name suggests household size; open: is the floor ok?
    "HHNBPERS_0_13_COD",  # name suggests children in the household; same open question
    "IN25",  # open: what does it count? is the floor ok?
    "IN26",  # open: what does it count? is the floor ok?
    "HO12_COD",  # open: count or nominal code? numeric may impose a false order
    "HO34_COD",  # open: count or nominal code? numeric may impose a false order
)

PARSED_COLUMNS = HHMM_COLUMNS + QUANTITY_COLUMNS + BUCKET_COLUMNS
FLAG_SUFFIX = "_FLAG"

_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")
_BUCKET = re.compile(r"^>=\s*(\d+)$")
_BUCKET_PT = re.compile(r"^(\d+)\s+ou\s+mais$")
_HHMM = re.compile(r"^\d{4}$")


def flag_column(column: str) -> str:
    """Name of the flag companion of ``column``."""
    return f"{column}{FLAG_SUFFIX}"


def _parse_number(value: str) -> float | None:
    """Return the number in ``value``, or ``None`` when it is not a number."""
    text = value.strip()
    if _NUMBER.match(text):
        return float(text)
    for pattern in (_BUCKET, _BUCKET_PT):
        match = pattern.match(text)
        if match:
            return float(match.group(1))
    return None


def _parse_hhmm(value: str) -> float | None:
    """Return ``60*HH + MM`` for a ``'HHMM'`` string, or ``None`` otherwise."""
    text = value.strip()
    if not _HHMM.match(text):
        return None
    hours, minutes = int(text[:2]), int(text[2:])
    if minutes > 59:
        raise ValueError(f"HHMM value {value!r} has minutes above 59")
    return float(60 * hours + minutes)


def _parse_column(series: pd.Series, parser, column: str) -> tuple[pd.Series, pd.Series]:
    """Parse one column and return ``(values, flags)``.

    ``values`` is float with NaN for every non-answer. ``flags`` is text with
    one level per case: the sentinel itself, ``"missing"`` for a real NaN, or
    ``"answer"``.
    """
    values: list[float | None] = []
    flags: list[str] = []
    unknown: set[str] = set()

    for raw in series:
        if pd.isna(raw):
            values.append(None)
            flags.append(FLAG_MISSING)
            continue
        text = str(raw).strip()
        if text in SENTINELS:
            values.append(None)
            flags.append(text)
            continue
        parsed = parser(text)
        if parsed is None:
            unknown.add(text)
            values.append(None)
            flags.append(FLAG_MISSING)
            continue
        values.append(parsed)
        flags.append(FLAG_ANSWER)

    if unknown:
        raise ValueError(
            f"column {column!r} has values that are neither numbers nor known "
            f"sentinels {SENTINELS}: {sorted(unknown)}"
        )
    return (
        pd.Series(values, index=series.index, dtype="float64"),
        pd.Series(flags, index=series.index, dtype="object"),
    )


def recode_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a recoded copy of ``frame``.

    Purpose:
        Parse the quantity, duration, and bucket columns to ``float`` and add a
        ``<COL>_FLAG`` column for each of them.

    Invariants:
        Pure and stateless: the input is never mutated, nothing is fitted, and
        there is no randomness. The same value always maps to the same output,
        whatever other rows are present, so this cannot leak.
        Original column names and order are preserved; flag columns are
        appended. Columns outside the rule-book are returned unchanged.
        Known columns that are absent from ``frame`` are skipped, so a small
        test frame works.

    Failure modes:
        ``ValueError`` when a parsed column contains a value that is neither a
        number nor a known sentinel (the message lists the values), or when an
        ``HHMM`` value is not four digits or has minutes above 59.
        Hours are not capped: values up to 30h exist and are real answers.
    """
    recoded = frame.copy()

    for column in PARSED_COLUMNS:
        if column not in recoded.columns:
            continue
        parser = _parse_hhmm if column in HHMM_COLUMNS else _parse_number
        values, flags = _parse_column(recoded[column], parser, column)
        recoded[column] = values
        recoded[flag_column(column)] = flags

    return recoded
