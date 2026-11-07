"""Run offline to regenerate every model and the untouched holdout results."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from ml_core import evaluate

ROOT = Path(__file__).resolve().parent


def main():
    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data
    # sklearn encodes malignant as 0; invert so clinical positive is malignant.
    y = (1 - dataset.target).rename('target')
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        'Logistic Regression': ('logistic_regression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42))),
        'Decision Tree': ('decision_tree', DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42)),
        'KNN': ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        'Naive Bayes': ('naive_bayes', GaussianNB()),
        'Support Vector Machine': ('svm', make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', random_state=42))),
    }
    (ROOT / 'model').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    rows, cv_rows = [], []
    for name, (slug, estimator) in models.items():
        cv = cross_validate(estimator, x_train, y_train,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring={'auc': 'roc_auc', 'f1': 'f1'}, n_jobs=1)
        cv_rows.append({'Model': name, 'CV AUC mean': np.mean(cv['test_auc']),
                        'CV AUC std': np.std(cv['test_auc']), 'CV F1 mean': np.mean(cv['test_f1'])})
        estimator.fit(x_train, y_train)
        joblib.dump(estimator, ROOT / 'model' / f'{slug}.joblib')
        rows.append({'Model': name, **evaluate(estimator, x_test, y_test)})
    pd.concat([x_test, y_test], axis=1).to_csv(ROOT / 'test_data.csv', index=False)
    pd.DataFrame(rows).to_csv(ROOT / 'results' / 'metrics.csv', index=False)
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(ROOT / 'results' / 'cross_validation.csv', index=False)
    metadata = {
        'features': list(x.columns), 'target': {'0': 'benign', '1': 'malignant'},
        'seed': 42, 'train_rows': len(x_train), 'test_rows': len(x_test),
        'train_class_counts': y_train.value_counts().sort_index().to_dict(),
        'test_class_counts': y_test.value_counts().sort_index().to_dict(),
        'sklearn_version': sklearn.__version__,
        'models': {name: f'{slug}.joblib' for name, (slug, _) in models.items()},
        'selected_by_training_cv_auc': str(cv_frame.loc[cv_frame['CV AUC mean'].idxmax(), 'Model']),
        'train_indices': x_train.index.tolist(), 'test_indices': x_test.index.tolist(),
    }
    (ROOT / 'model' / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print('Selected using training-only CV AUC:', metadata['selected_by_training_cv_auc'])


if __name__ == '__main__':
    main()
