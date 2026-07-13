#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# SeedBrain Release Script
# Builds model, packages release artifacts, creates git tag.
# Usage: ./release.sh [VERSION_TAG]
#   Default tag: CRS_V2.5_Lite-$(date +%Y%m%d)
# ============================================================

TAG="${1:-CRS_V2.5_Lite-$(date +%Y%m%d)}"
DIR="CRS_V2.5_Lite"

if [ ! -d "$DIR" ]; then
    echo "Error: $DIR/ not found. Run from repo root."
    exit 1
fi

echo "==> Creating venv and installing deps..."
python3 -m venv /tmp/seedbrain_venv
source /tmp/seedbrain_venv/bin/activate
pip install -q -r "$DIR/requirements.txt"

echo "==> Training model..."
cd "$DIR"
python -c "
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
crop_ranges = {}
for crop in sorted(df['label'].unique()):
    sub = df[df['label'] == crop]
    crop_ranges[crop] = {
        'N': [int(sub['N'].min()), int(sub['N'].max())],
        'P': [int(sub['P'].min()), int(sub['P'].max())],
        'K': [int(sub['K'].min()), int(sub['K'].max())],
        'temperature': [round(float(sub['temperature'].min()),1), round(float(sub['temperature'].max()),1)],
        'humidity': [round(float(sub['humidity'].min()),1), round(float(sub['humidity'].max()),1)],
        'ph': [round(float(sub['ph'].min()),2), round(float(sub['ph'].max()),2)],
        'rainfall': [round(float(sub['rainfall'].min()),1), round(float(sub['rainfall'].max()),1)],
    }
with open(BASE / 'crop_ranges.json', 'w') as f:
    json.dump(crop_ranges, f, indent=2)
acc = model.score(X_val, y_val)
print(f'Validation accuracy: {acc:.4f}')
"
cd ..

echo "==> Pinning dependencies..."
pip freeze --exclude-editable > "$DIR/requirements.lock"

echo "==> Package size: $(du -sh $DIR/ | cut -f1)"

echo "==> Creating release archive..."
cd /tmp
tar czf "seedbrain-${TAG}.tar.gz" "$OLDPWD/$DIR/"
zip -r "seedbrain-${TAG}.zip" "$OLDPWD/$DIR/"
cd "$OLDPWD"

echo "==> Checksums..."
sha256sum "/tmp/seedbrain-${TAG}.tar.gz" "/tmp/seedbrain-${TAG}.zip"

echo ""
echo "============================================"
echo "Release artifacts created:"
echo "  /tmp/seedbrain-${TAG}.tar.gz"
echo "  /tmp/seedbrain-${TAG}.zip"
echo ""
echo "To create GitHub release, run:"
echo "  gh release create $TAG \\"
echo "    /tmp/seedbrain-${TAG}.tar.gz \\"
echo "    /tmp/seedbrain-${TAG}.zip \\"
echo "    --title 'SeedBrain $TAG' \\"
echo "    --notes 'See CHANGELOG'"
echo "============================================"

deactivate
rm -rf /tmp/seedbrain_venv
