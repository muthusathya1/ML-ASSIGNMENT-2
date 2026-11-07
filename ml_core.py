"""Shared input contract and binary classification metrics (malignant=1)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def validate_data(frame, features):
    if frame.empty:
        raise ValueError('The CSV must contain at least one data row.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    expected = set(features) | {'target'}
    missing, extra = expected - set(frame.columns), set(frame.columns) - expected
    if missing or extra:
        raise ValueError(f'CSV schema mismatch. Missing: {sorted(missing)}; unexpected: {sorted(extra)}')
    try:
        x = frame.loc[:, features].apply(pd.to_numeric, errors='raise')
        y = pd.to_numeric(frame['target'], errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Features and target must be numeric.') from exc
    if not np.isfinite(x.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError('Missing values and infinity are not allowed.')
    if not y.isin([0, 1]).all():
        raise ValueError('Target must be 0 (benign) or 1 (malignant).')
    return x, y.astype(int)


def evaluate(model, x, y):
    predicted = model.predict(x)
    if hasattr(model, 'predict_proba'):
        positive_index = list(model.classes_).index(1)
        score = model.predict_proba(x)[:, positive_index]
    else:
        score = model.decision_function(x)
    return {
        'Accuracy': accuracy_score(y, predicted),
        'AUC': roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        'Precision': precision_score(y, predicted, zero_division=0),
        'Recall': recall_score(y, predicted, zero_division=0),
        'F1': f1_score(y, predicted, zero_division=0),
        'MCC': matthews_corrcoef(y, predicted),
    }
