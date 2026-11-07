"""Interactive evaluation only: this app never fits a model."""
import csv
import io
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from ml_core import validate_data, evaluate

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='CellScope | Classifier comparison', page_icon='🔬', layout='wide')


@st.cache_resource
def load_artifacts():
    metadata = json.loads((ROOT / 'model' / 'metadata.json').read_text(encoding='utf-8'))
    models = {name: joblib.load(ROOT / 'model' / filename) for name, filename in metadata['models'].items()}
    return metadata, models


st.title('🔬 CellScope')
st.write('Breast Cancer Wisconsin Diagnostic · Five perspectives on the same cells')
st.caption('Educational model comparison. Positive class: malignant (1); negative class: benign (0).')
try:
    metadata, models = load_artifacts()
except (OSError, ValueError) as exc:
    st.error(f'Model artifacts unavailable. Run python train.py first. Details: {exc}')
    st.stop()

with st.sidebar:
    st.header('Evaluation controls')
    selected = st.selectbox('Choose a classifier', list(models))
    uploaded = st.file_uploader('Upload test data (CSV)', type=['csv'])
    st.download_button('Download held-out test CSV', (ROOT / 'test_data.csv').read_bytes(), 'test_data.csv', 'text/csv')
    st.caption('CSV must contain the 30 original feature columns and target (0 or 1). Only upload held-out test data.')
    with st.expander('Required feature names'):
        st.write(metadata['features'])

try:
    if uploaded is not None:
        text = uploaded.getvalue().decode('utf-8-sig')
        header = next(csv.reader(io.StringIO(text)), [])
        if len(header) != len(set(header)):
            raise ValueError('Duplicate CSV headers are not allowed.')
        frame = pd.read_csv(io.StringIO(text))
    else:
        frame = pd.read_csv(ROOT / 'test_data.csv')
    x, y = validate_data(frame, metadata['features'])
except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(f'Cannot evaluate this CSV: {exc}')
    st.stop()

st.info(f"{'Uploaded' if uploaded is not None else 'Bundled held-out'} test data · {len(x)} rows · {x.shape[1]} features · {int(y.sum())} malignant cases")
if y.nunique() < 2:
    st.warning('Only one target class is present. AUC is unavailable; other metrics have limited interpretation.')
st.subheader(selected)
scores = evaluate(models[selected], x, y)
for column, (name, value) in zip(st.columns(6), scores.items()):
    column.metric(name, 'N/A' if pd.isna(value) else f'{value:.4f}')

comparison = pd.DataFrame([{'Model': name, **evaluate(model, x, y)} for name, model in models.items()]).set_index('Model')
st.subheader('All five models · same test data')
st.dataframe(
    comparison,
    column_config={name: st.column_config.NumberColumn(format='%.4f') for name in comparison.columns},
    width='stretch',
)
st.download_button('Download comparison results', comparison.to_csv().encode(), 'comparison.csv', 'text/csv')

left, right = st.columns(2)
predicted = models[selected].predict(x)
with left:
    st.subheader('Confusion matrix')
    st.caption('Rows: actual class · Columns: predicted class')
    st.dataframe(pd.DataFrame(confusion_matrix(y, predicted, labels=[0, 1]),
                              index=['Actual benign', 'Actual malignant'],
                              columns=['Predicted benign', 'Predicted malignant']), width='stretch')
with right:
    st.subheader('Classification report')
    report = classification_report(y, predicted, labels=[0, 1], target_names=['Benign', 'Malignant'], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, width='stretch')

with st.expander('Inspect data and predictions'):
    result = frame.copy()
    result['predicted_target'] = predicted
    st.dataframe(result, width='stretch')
    st.download_button('Download predictions', result.to_csv(index=False).encode(), 'predictions.csv', 'text/csv')
with st.expander('How this experiment works'):
    st.write('569 original records; stratified 80/20 split, seed 42. Scaling is fitted only inside training pipelines. Five-fold stratified training-only cross-validation provides a separate model-selection check. Hyperparameters are fixed baselines. Uploaded data is evaluated with saved models; no retraining occurs.')
    st.write('AUC uses malignant probabilities or SVM decision scores. Precision, recall and F1 use malignant as the positive class. Undefined precision/recall/F1 are reported as zero; single-class AUC is unavailable.')
    st.link_button('Original UCI dataset', 'https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic')
