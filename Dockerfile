FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY CRS_V2.5_Lite/requirements.txt CRS_V2.5_Lite/requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY CRS_V2.5_Lite/ .

RUN python -c "
import pandas as pd, numpy as np, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings('ignore')

BASE = Path('.')
df = pd.read_csv(BASE / 'dataset' / 'Crop_recommendation.csv')
df['label'] = df['label'].str.strip().str.lower()
FEATS = ['N','P','K','temperature','humidity','ph','rainfall']
X = df[FEATS].values
le = LabelEncoder()
y = le.fit_transform(df['label'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_lambda=1.0, reg_alpha=0.5,
    random_state=42, eval_metric='mlogloss', verbosity=0)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
joblib.dump(model, BASE / 'xgboost_model.pkl')
joblib.dump(le, BASE / 'label_encoder.pkl')
print(f'Validation accuracy: {model.score(X_val, y_val):.4f}')
"

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
