#!/usr/bin/env python3
"""Train XGBoost model on merged dataset and save artifacts."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES = ['N', 'P', 'K', 'organic_carbon', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET = 'label'

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5


def load_merged_data():
    path = BASE_DIR / 'dataset' / 'gov_dataset.csv'
    if not path.exists():
        path = BASE_DIR / 'dataset' / 'Crop_recommendation_merged.csv'
    df = pd.read_csv(path)
    df['label'] = df['label'].str.strip().str.lower()
    return df


def train_xgboost(X_train, y_train, X_val, y_val, cv_mode=False):
    model = XGBClassifier(
        n_estimators=500,
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
        use_label_encoder=False,
        verbosity=0,
    )
    if not cv_mode:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def main():
    print("=" * 60)
    print("MODEL TRAINING (8 raw features, no interactions)")
    print("=" * 60)

    print("\n1. Loading merged dataset...")
    df = load_merged_data()
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"   {n_before} rows → {len(df)} unique, {df['label'].nunique()} crops")

    X = df[FEATURES].values
    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    print(f"\n2. Train/val split (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")

    print(f"\n3. Training XGBoost (8 features, no interaction features)...")
    model = train_xgboost(X_train, y_train, X_val, y_val)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"   Train accuracy: {train_acc:.4f}")
    print(f"   Val accuracy:   {val_acc:.4f}")

    print(f"\n4. Cross-validation ({N_FOLDS}-fold) on all data...")
    cv_model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_lambda=1.0, reg_alpha=0.5,
        random_state=RANDOM_STATE, eval_metric='mlogloss',
        use_label_encoder=False, verbosity=0,
    )
    cv_scores = cross_val_score(cv_model, X, y, cv=N_FOLDS, scoring='accuracy')
    print(f"   CV scores: {cv_scores}")
    print(f"   CV mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print(f"\n5. Per-class metrics (on clean validation set):")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    print(f"\n6. Saving model...")
    model_path = BASE_DIR / 'xgboost_model.pkl'
    joblib.dump(model, model_path)
    print(f"   → Saved {model_path}")

    le_path = BASE_DIR / 'label_encoder.pkl'
    joblib.dump(le, le_path)
    print(f"   → Saved {le_path}")

    print(f"\n7. Crop label mapping (unique samples):")
    for i, label in enumerate(le.classes_):
        count = len(df[df['label'] == label])
        print(f"   {i}: {label} ({count} samples)")

    print(f"\n8. Feature importance:")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importance), key=lambda x: -x[1]):
        print(f"   {feat}: {imp:.4f}")

    print(f"\n✅ Training complete! CV accuracy: {cv_scores.mean():.4f} "
          f"(val: {val_acc:.4f})")
    return model, le, val_acc


if __name__ == "__main__":
    main()
