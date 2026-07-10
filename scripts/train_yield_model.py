#!/usr/bin/env python3
"""Train Africa yield prediction model (single XGBoost with crop as feature)."""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES = ['N', 'P', 'K', 'organic_carbon', 'ph', 'temperature', 'humidity', 'rainfall']
RANDOM_STATE = 42


def load_data():
    path = BASE_DIR / 'dataset' / 'grow_dataset.csv'
    df = pd.read_csv(path)
    df['crop'] = df['crop'].str.strip().str.lower()
    df = df.dropna(subset=FEATURES + ['yield_ton_ha'])
    return df


def main():
    print("=" * 65)
    print("AFRICA YIELD PREDICTION (single model, crop as feature)")
    print("=" * 65)

    print("\n1. Loading GROW-Africa dataset...")
    df = load_data()
    print(f"   {len(df)} rows, {df['crop'].nunique()} crops")

    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['crop'])
    all_features = FEATURES + ['crop_encoded']

    X = df[all_features].values
    y = df['yield_ton_ha'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\n2. Training XGBoost regressor ({len(all_features)} features)...")
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=1.0,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    val_mae = mean_absolute_error(y_val, y_pred_val)

    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Val R²:   {val_r2:.4f}")
    print(f"   Val MAE:  {val_mae:.4f} ton/ha")

    print(f"\n3. Per-crop validation metrics:")
    val_df = pd.DataFrame({'crop': le.inverse_transform(X_val[:, -1].astype(int)),
                           'actual': y_val, 'predicted': y_pred_val})
    per_crop = []
    for crop in sorted(val_df['crop'].unique()):
        sub = val_df[val_df['crop'] == crop]
        r2 = r2_score(sub['actual'], sub['predicted'])
        mae = mean_absolute_error(sub['actual'], sub['predicted'])
        per_crop.append({'crop': crop, 'n': len(sub), 'r2': round(r2, 4),
                         'mae': round(mae, 4)})
    per_crop_df = pd.DataFrame(per_crop)
    for _, r in per_crop_df.iterrows():
        print(f"   {r['crop']:15s}: n={r['n']:5d}  val_r2={r['r2']:.3f}  val_mae={r['mae']:.3f}")

    print(f"\n4. Saving...")
    artifacts = {'model': model, 'label_encoder': le, 'features': all_features}
    path = BASE_DIR / 'africa_yield_model.pkl'
    joblib.dump(artifacts, path)
    print(f"   → Saved {path}")

    crop_list = sorted(le.classes_)
    with open(BASE_DIR / 'africa_crops.json', 'w') as f:
        json.dump(crop_list, f)
    print(f"   → Saved africa_crops.json ({len(crop_list)} crops)")

    print(f"\n5. Feature importance:")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(all_features, importance), key=lambda x: -x[1]):
        print(f"   {feat}: {imp:.4f}")

    print(f"\n✅ Training complete! Val R²: {val_r2:.4f}, MAE: {val_mae:.4f} ton/ha")


if __name__ == "__main__":
    main()
