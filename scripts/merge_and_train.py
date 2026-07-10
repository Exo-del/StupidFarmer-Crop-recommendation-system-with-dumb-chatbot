#!/usr/bin/env python3
"""Merge GAEZ global + GROW-Africa into one unified dataset, train single classifier."""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES = ['N', 'P', 'K', 'organic_carbon', 'temperature', 'humidity', 'ph', 'rainfall']
RANDOM_STATE = 42


def load_and_merge():
    gov = pd.read_csv(BASE_DIR / 'dataset' / 'gov_dataset.csv')
    grow = pd.read_csv(BASE_DIR / 'dataset' / 'grow_dataset.csv')

    # Standardize column names
    gov = gov[FEATURES + ['continent', 'label']].copy()
    gov['source'] = 'gaez'

    grow = grow.rename(columns={'crop': 'label'})
    grow['continent'] = 'africa'
    grow['source'] = 'grow_africa'
    grow = grow[FEATURES + ['continent', 'label', 'source']].copy()

    # Normalize labels
    gov['label'] = gov['label'].str.strip().str.lower()
    grow['label'] = grow['label'].str.strip().str.lower()

    # Map crop names to a common set
    name_map = {
        'sugar cane': 'sugarcane',
        'sweet potato': 'sweet_potato',
        'pigeon pea': 'pigeonpea',
    }
    gov['label'] = gov['label'].replace(name_map)
    grow['label'] = grow['label'].replace(name_map)

    df = pd.concat([gov, grow], ignore_index=True)
    df = df.dropna(subset=FEATURES + ['label'])
    df = df.drop_duplicates(subset=FEATURES + ['continent', 'label'])

    return df


def main():
    print("=" * 65)
    print("UNIFIED CROP RECOMMENDATION MODEL")
    print("(GAEZ global + GROW-Africa merged)")
    print("=" * 65)

    print("\n1. Loading and merging datasets...")
    df = load_and_merge()
    print(f"   Total: {len(df)} rows")
    print(f"   Sources: {df['source'].value_counts().to_dict()}")
    print(f"   Continents: {df['continent'].value_counts().to_dict()}")
    print(f"   Crops: {df['label'].nunique()}")

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    continent_le = LabelEncoder()
    df['continent_encoded'] = continent_le.fit_transform(df['continent'])

    all_features = FEATURES + ['continent_encoded']
    X = df[all_features].values
    y = df['label_encoded'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\n2. Training XGBoost classifier ({len(all_features)} features)...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=1.0,
        reg_alpha=0.5,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        verbosity=0,
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"   Train accuracy: {train_acc:.4f}")
    print(f"   Val accuracy:   {val_acc:.4f}")

    print(f"\n3. Per-class metrics:")
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=le.classes_, digits=3)
    print(report)

    print(f"\n4. Saving unified model...")
    joblib.dump(model, BASE_DIR / 'xgboost_model.pkl')
    joblib.dump(le, BASE_DIR / 'label_encoder.pkl')
    joblib.dump(continent_le, BASE_DIR / 'continent_encoder.pkl')
    print(f"   → Saved xgboost_model.pkl + label_encoder.pkl + continent_encoder.pkl")

    crops_list = sorted(le.classes_.tolist())
    with open(BASE_DIR / 'model_crops.json', 'w') as f:
        json.dump(crops_list, f)
    print(f"   → Saved model_crops.json ({len(crops_list)} crops)")

    print(f"\n5. Feature importance:")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(all_features, importance), key=lambda x: -x[1]):
        print(f"   {feat}: {imp:.4f}")

    print(f"\n✅ Done! Val accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
