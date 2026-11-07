"""Deterministic source-data cleaning and descriptive training-only outlier audit."""
import numpy as np
import pandas as pd
from ml_core import validate_data


def clean_training_data(frame, features):
    """Validate source records and remove exact repeats before splitting.

    Missing/invalid values fail explicitly rather than being silently discarded.
    The supplied UCI copy has no missing values, so imputation is unnecessary.
    Conflicting labels for identical measurements require investigation.
    """
    x, y = validate_data(frame, features)
    normalized = x.copy()
    normalized['target'] = y
    cleaned = normalized.drop_duplicates(keep='first').copy()
    if cleaned.duplicated(subset=features, keep=False).any():
        raise ValueError('Identical feature rows have conflicting target labels; investigate the source data.')
    audit = {
        'rows_before_cleaning': int(len(frame)),
        'feature_count': len(features),
        'missing_values_total': int(frame.isna().sum().sum()),
        'missing_values_by_column': {column: int(count) for column, count in frame.isna().sum().items()},
        'nonfinite_values_total': int((~np.isfinite(normalized.to_numpy(dtype=float))).sum()),
        'duplicate_feature_rows_before_cleaning': int(x.duplicated().sum()),
        'duplicate_rows_removed': int(len(normalized) - len(cleaned)),
        'rows_after_cleaning': int(len(cleaned)),
        'class_counts_after_cleaning': {str(label): int(count) for label, count in cleaned['target'].value_counts().sort_index().items()},
        'missing_value_policy': 'Reject missing values explicitly; no imputation needed for this complete dataset.',
        'duplicate_policy': 'Keep first occurrence of each exact labelled record before splitting; reject conflicting labels.',
        'invalid_value_policy': 'Require numeric finite features and target in {0,1}; reject invalid records.',
    }
    return cleaned, audit


def inspect_training_outliers(x_train):
    """Flag outside Q1-1.5*IQR / Q3+1.5*IQR; never remove or clip rows.

    Thresholds use training features only, never holdout data or target labels.
    An IQR flag is descriptive evidence, not proof of an erroneous measurement.
    """
    q1, q3 = x_train.quantile(.25), x_train.quantile(.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = x_train.lt(lower) | x_train.gt(upper)
    report = pd.DataFrame({
        'feature': x_train.columns,
        'q1': q1.to_numpy(), 'q3': q3.to_numpy(), 'iqr': iqr.to_numpy(),
        'lower_fence': lower.to_numpy(), 'upper_fence': upper.to_numpy(),
        'outlier_count': flagged.sum().to_numpy(),
        'outlier_fraction': flagged.mean().to_numpy(),
    })
    summary = {
        'scope': 'Training features only; 1.5 x IQR rule per feature.',
        'training_rows_inspected': int(len(x_train)),
        'rows_with_any_outlier': int(flagged.any(axis=1).sum()),
        'flagged_feature_values': int(flagged.sum().sum()),
        'constant_features': x_train.columns[x_train.nunique() <= 1].tolist(),
        'rows_removed': 0,
        'policy': 'Retain all flagged measurements; extreme cell measurements may carry classification signal. No clipping or outlier-based tuning.',
    }
    return report, summary
