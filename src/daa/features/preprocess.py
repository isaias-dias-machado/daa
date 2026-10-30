"""The feature boundary: a ``ColumnTransformer`` built from a ``ColumnSpec``.

Everything here must be fitted inside a pipeline (and therefore inside a CV
fold). Fitting a preprocessor on the full dataset before cross-validation leaks
validation information; that is the bug the primer notebook demonstrates.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ..data import ColumnSpec

VARIANTS = ("impute", "passthrough")


def build_preprocessor(spec: ColumnSpec, variant: str = "impute") -> ColumnTransformer:
    """Return an unfitted preprocessor for the columns in ``spec``.

    Contract:
        Input: a raw ``DataFrame`` containing the columns named in ``spec``.
        Output: a named ``DataFrame`` whose columns are
        ``get_feature_names_out()``.

    Variants:
        ``"impute"``: median-impute and scale numeric columns, one-hot encode
        nominal columns, most-frequent-impute and ordinal-encode ordinal
        columns. The output contains no NaN.
        ``"passthrough"``: leave numeric NaN in place for models that support
        them (``HistGradientBoostingClassifier``); still encode categoricals.

    Failure modes:
        ``ValueError`` for an unknown variant or a spec with no input columns.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if not spec.input_columns:
        raise ValueError("ColumnSpec has no input columns")

    transformers: list[tuple[str, object, list[str]]] = []

    if spec.numeric:
        numeric: object
        if variant == "impute":
            numeric = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            )
        else:
            numeric = "passthrough"
        transformers.append(("numeric", numeric, list(spec.numeric)))

    if spec.nominal:
        transformers.append(
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(spec.nominal),
            )
        )

    if spec.ordinal:
        ordinal = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("ordinal", ordinal, list(spec.ordinal)))

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")
