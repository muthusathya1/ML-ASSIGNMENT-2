# CellScope - ML Assignment 2

## a. Problem statement

Compare five classification algorithms for predicting whether a breast mass is benign or malignant from cell-nucleus measurements. Evaluate every model on the same held-out samples, and provide a Streamlit interface for evaluating a labelled test CSV using saved models. This is an educational experiment, not a clinical tool.

## b. Dataset description

[UCI Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic), DOI: 10.24432/C5DW2B. Creators: William Wolberg, Olvi Mangasarian and W. Nick Street. UCI lists a CC BY 4.0 license.

The scikit-learn bundled copy supplies 569 instances and 30 numeric predictors, satisfying the assignment minimums (500 instances and 12 features). Features describe mean, standard error and worst values for ten characteristics of cell nuclei extracted from digitized fine-needle aspirate images. ID is not used. There are no missing feature values in this copy.

There are 357 benign and 212 malignant samples. **This project recodes target as 0=benign, 1=malignant**, reversing the scikit-learn built-in target encoding. Uploaded CSVs must use this project's encoding.

### Experimental method

- Stratified 80/20 split with `random_state=42`: 455 training and 114 test samples. Test set contains 72 benign and 42 malignant cases.
- StandardScaler is inside the Logistic Regression, KNN and SVM pipelines. It is fitted on training samples only; Decision Tree and GaussianNB use original numeric measurements.
- Fixed baseline settings: Logistic Regression (max_iter=3000); Decision Tree (max_depth=5, min_samples_leaf=3); KNN (k=5); GaussianNB (defaults); SVM (RBF, C=1, gamma=scale).
- Five-fold stratified cross-validation uses **training data only**, with scaling refitted inside each fold. Models are subsequently fitted on all 455 training rows and evaluated once on the 114 test rows. No hyperparameter search or test-driven tuning is performed.
- Precision, recall and F1 are binary scores for malignant=1. AUC uses continuous malignant probabilities, or the SVM decision function. MCC and accuracy use predicted classes.
- Undefined precision/recall/F1 use zero. Single-class uploads display AUC as N/A. Such uploads are insufficient for a meaningful full comparison.

### Data cleaning and quality checks

`train.py` runs the checks in `data_quality.py` before the stratified split. It validates the numeric feature schema, finite values and binary labels, then removes exact duplicate labelled records while preserving the first occurrence. Identical feature records with conflicting labels cause an explicit error for investigation. Removing repeats before splitting prevents identical records from leaking across train and test sets.

| Check | Result | Action |
|---|---:|---|
| Source records | 569 | All retained |
| Missing values (features and target) | 0 | No imputation needed |
| Infinite or invalid numeric values | 0 | Validation passed |
| Duplicate feature rows | 0 | No repeated measurements found |
| Exact duplicate labelled records | 0 | Deduplication removed 0 rows |
| Constant training features | 0 | No features removed |
| Training rows with at least one IQR outlier flag | 135 of 455 | Retained for modeling |
| Flagged training feature values | 494 | Inspected through per-feature summary |

Missing or invalid values cause training to stop with an error rather than silently deleting observations. Since this dataset is complete, an imputer is unnecessary. If another dataset required imputation, its parameters would need to be fitted inside each training pipeline and cross-validation fold.

Outlier inspection uses **training features only**: values below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR are flagged separately for each feature. Thresholds never use the test set or target labels. An IQR flag does not establish a data error; extreme cell measurements may be useful classification signals. No outlier removal or clipping is applied. StandardScaler subsequently fits on the training data inside the relevant model pipelines.

The reproducible audit is saved in `results/data_quality.json`; quartiles, fences, counts and proportions for all 30 training features are saved in `results/training_outliers.csv`. Uploaded evaluation CSVs remain subject to strict validation and are evaluated as supplied, without silently removing duplicate test rows. The existing 455/114 split and all reported metrics are unchanged after running the audit.

## c. GitHub Repository Link

[GitHub repository](https://github.com/muthusathya1/ML-ASSIGNMENT-2)

[Live Streamlit app](https://muthusathya1-ml-assignment-2-app-l7dhor.streamlit.app/)

## d. Models used

Results below are calculated by `python train.py`, not illustrative values. Full-precision outputs are in `results/metrics.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.964912 | 0.996032 | 0.975000 | 0.928571 | 0.951220 | 0.924518 |
| Decision Tree | 0.903509 | 0.947090 | 0.942857 | 0.785714 | 0.857143 | 0.792640 |
| KNN | 0.956140 | 0.982308 | 0.974359 | 0.904762 | 0.938272 | 0.905824 |
| Naive Bayes | 0.938596 | 0.993386 | 1.000000 | 0.833333 | 0.909091 | 0.871489 |
| Support Vector Machine | 0.973684 | 0.994709 | 1.000000 | 0.928571 | 0.962963 | 0.944155 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | 110/114 correct; 3 malignant cases missed and 1 benign case flagged. Highest holdout AUC (0.9960). Scaling and a linear boundary work well for these measurements. It also has the highest mean training CV AUC (0.9958). |
| Decision Tree | 103/114 correct; 9 malignant cases missed and 2 benign cases flagged. Lowest recall and MCC here. Its fixed depth and leaf constraints limit complexity; threshold-based splits may lose information in correlated continuous features. This is a plausible interpretation, not a measured causal conclusion. |
| KNN | 109/114 correct; 4 malignant cases missed and 1 benign case flagged. Scaled distance-based classification is competitive, but less sensitive than Logistic Regression and SVM on this split. |
| Naive Bayes | 107/114 correct; 7 malignant cases missed, no false positives. Perfect precision on this small test set does not mean perfect detection: recall is 0.8333. Its independence assumption is restrictive for related cell measurements. |
| Support Vector Machine | 111/114 correct; 3 malignant cases missed, no false positives. Highest observed accuracy, F1 and MCC, with recall equal to Logistic Regression. The scaled RBF boundary performs well on this holdout. |
| Overall winner for this dataset? | SVM wins this holdout comparison on accuracy/F1/MCC. For selection without consulting test results, Logistic Regression wins the predefined training CV AUC criterion (0.9958 vs SVM 0.9949). These are different criteria; the small differences on one split do not establish general superiority. |

Training-only CV results (fold means; AUC standard deviations also saved in `results/cross_validation.csv`):

| Model | Mean CV AUC | Mean CV F1 |
|---|---:|---:|
| Logistic Regression | 0.995769 | 0.964048 |
| Decision Tree | 0.935552 | 0.896334 |
| KNN | 0.987100 | 0.951042 |
| Naive Bayes | 0.987822 | 0.918514 |
| Support Vector Machine | 0.994943 | 0.960987 |

## Run locally

Use Python 3.12. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501). The app initially evaluates the included held-out CSV. Download that CSV, upload it, select each classifier, and inspect all six metrics, the comparison table, confusion matrix and classification report. Downloads provide comparisons and row-level predictions.

### CSV contract

Use the exact 30 feature headers in `test_data.csv` plus `target`. Columns may be reordered. Extra/missing/duplicate columns, empty data, nonnumeric values, NaN/infinity and target values outside {0,1} are rejected. The app accepts UTF-8 CSVs. It never calls `fit` or trains on an upload.

### Project files

- `train.py`: offline training, cross-validation, held-out export and model persistence.
- `ml_core.py`: shared validation and metric calculation.
- `data_quality.py`: source validation, duplicate removal and training-only outlier inspection.
- `app.py`: interactive saved-model evaluation.
- `model/`: five joblib pipelines and metadata containing feature names, versions and split row indices.
- `test_data.csv`: 114 held-out rows with target.
- `results/`: computed holdout metrics, training cross-validation metrics, data-quality audit and per-feature training outlier summary.

Saved scikit-learn artifacts require compatible dependencies; use pinned requirements and regenerate artifacts after changing scikit-learn. Only load trusted joblib files. Models are loaded from this project's model directory, never from uploaded files.

## Remaining submission steps

1. Review the experimental results and model explanations.
2. Execute the project on BITS Virtual Lab and capture an authentic screenshot. Local execution does not satisfy that requirement.
3. GitHub repository and live Streamlit app links are provided in section c above.
4. Verify CSV upload and all model selections in the deployed app.
5. Prepare the final single PDF in required order: GitHub link, live app link, BITS execution screenshot, full README content. The final submission PDF still requires authentic BITS execution evidence.
6. Submit before 8 November 2026, 23:59 (per assignment brief).
