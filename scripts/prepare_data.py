#!/usr/bin/env python3
"""Normalize, merge, and prepare datasets for training."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import shutil
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / 'dataset' / 'raw'
OUTPUT_DIR = BASE_DIR / 'dataset'

MIN_SAMPLES_PER_CROP = 200

ZENODO_PATH = RAW_DIR / 'zenodo_original.csv'
SF24_PATH = Path('/home/louhmiwaslost/.cache/kagglehub/datasets/datasetengineer/smart-farming-data-2024-sf24/versions/1/Crop_recommendationV2.csv')
KAGGLE_PATH = BASE_DIR / 'dataset' / 'Crop_recommendation.csv'

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET = 'label'


def load_zenodo(path):
    df = pd.read_csv(path)
    df.rename(columns={'crop_label': 'label'}, inplace=True)
    df['label'] = df['label'].str.strip().str.lower()
    if 'soil' in df.columns:
        df['soil'] = df['soil'].str.strip().str.lower()
    for col in FEATURES:
        if col not in df.columns:
            alt = col.lower()
            if alt in df.columns:
                df.rename(columns={alt: col}, inplace=True)
    cols = [c for c in FEATURES + ['label', 'soil'] if c in df.columns]
    return df[cols].copy()


def load_kaggle(path):
    df = pd.read_csv(path)
    df['label'] = df['label'].str.strip().str.lower()
    for col in FEATURES:
        if col not in df.columns:
            alt = col.lower()
            if alt in df.columns:
                df.rename(columns={alt: col}, inplace=True)
    return df[FEATURES + ['label']].copy()


def load_sf24(path):
    df = pd.read_csv(path)
    df['label'] = df['label'].str.strip().str.lower()
    if 'Crop' in df.columns:
        df.rename(columns={'Crop': 'label'}, inplace=True)
    for col in FEATURES:
        if col not in df.columns:
            alt = col.lower()
            if alt in df.columns:
                df.rename(columns={alt: col}, inplace=True)
    cols = [c for c in FEATURES + ['label', 'soil_type', 'soil_moisture', 'organic_matter'] if c in df.columns]
    core = df[cols].copy()
    if 'soil_type' in core.columns:
        core.rename(columns={'soil_type': 'soil_type_code'}, inplace=True)
    return core


def compute_soil_profiles(zenodo_df):
    """Compute NPK/pH profiles per soil type from Zenodo real data."""
    profiles = {}
    for soil_type in sorted(zenodo_df['soil'].dropna().unique()):
        sub = zenodo_df[zenodo_df['soil'] == soil_type]
        profiles[soil_type] = {
            'N_median': round(float(sub['N'].median()), 1),
            'N_low': int(sub['N'].quantile(0.1)),
            'N_high': int(sub['N'].quantile(0.9)),
            'P_median': round(float(sub['P'].median()), 1),
            'P_low': int(sub['P'].quantile(0.1)),
            'P_high': int(sub['P'].quantile(0.9)),
            'K_median': round(float(sub['K'].median()), 1),
            'K_low': int(sub['K'].quantile(0.1)),
            'K_high': int(sub['K'].quantile(0.9)),
            'ph_median': round(float(sub['ph'].median()), 2),
            'ph_low': round(float(sub['ph'].quantile(0.1)), 2),
            'ph_high': round(float(sub['ph'].quantile(0.9)), 2),
            'samples': len(sub),
        }
    return profiles


def build_crop_region_map(zenodo_crops, kaggle_crops):
    """Determine which crops grow in which regions based on dataset origin."""
    region_map = {}
    for c in zenodo_crops:
        region_map[c] = ['africa']
    for c in kaggle_crops:
        if c in region_map:
            region_map[c].append('asia')
        else:
            region_map[c] = ['asia']
    return {c: sorted(set(r)) for c, r in region_map.items()}


def build_val_ranges(df):
    """Compute validation/input ranges from the merged dataset."""
    ranges = {}
    for feat in FEATURES:
        ranges[feat] = {
            'min': float(df[feat].min()),
            'max': float(df[feat].max()),
            'q01': float(df[feat].quantile(0.01)),
            'q99': float(df[feat].quantile(0.99)),
        }
    return ranges


def build_crop_ranges(df):
    """Build crop_ranges.json from merged dataset."""
    crop_ranges = {}
    for crop in sorted(df['label'].unique()):
        sub = df[df['label'] == crop]
        crop_ranges[crop] = {
            'N': [int(sub['N'].min()), int(sub['N'].max())],
            'P': [int(sub['P'].min()), int(sub['P'].max())],
            'K': [int(sub['K'].min()), int(sub['K'].max())],
            'temperature': [round(float(sub['temperature'].min()), 1), round(float(sub['temperature'].max()), 1)],
            'humidity': [round(float(sub['humidity'].min()), 1), round(float(sub['humidity'].max()), 1)],
            'ph': [round(float(sub['ph'].min()), 2), round(float(sub['ph'].max()), 2)],
            'rainfall': [round(float(sub['rainfall'].min()), 1), round(float(sub['rainfall'].max()), 1)],
        }
    return crop_ranges


def main():
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    print("\n1. Loading datasets...")
    zn = load_zenodo(ZENODO_PATH)
    print(f"   Zenodo (Africa): {len(zn)} rows, {zn['label'].nunique()} crops")
    print(f"     Crops: {sorted(zn['label'].unique())}")

    kg = load_kaggle(KAGGLE_PATH)
    print(f"   Kaggle (Asia):   {len(kg)} rows, {kg['label'].nunique()} crops")

    sf24 = load_sf24(SF24_PATH)
    print(f"   SF24 (Asia*):    {len(sf24)} rows, {sf24['label'].nunique()} crops")
    print(f"     * SF24 has same crop data as Kaggle, just extra columns")

    print("\n2. Computing soil type profiles from Zenodo data...")
    soil_profiles = compute_soil_profiles(zn)
    for st, prof in soil_profiles.items():
        print(f"   {st}: N={prof['N_median']} P={prof['P_median']} K={prof['K_median']} "
              f"pH={prof['ph_median']} ({prof['samples']} samples)")

    with open(BASE_DIR / 'soil_profiles.json', 'w') as f:
        json.dump(soil_profiles, f, indent=2)
    print("   → Saved soil_profiles.json")

    print("\n3. Merging datasets...")
    zn_core = zn[FEATURES + ['label']].copy()
    zn_core['source'] = 'zenodo'

    kg_core = kg[FEATURES + ['label']].copy()
    kg_core['source'] = 'kaggle'

    merged = pd.concat([zn_core, kg_core], ignore_index=True)
    print(f"   Merged: {len(merged)} rows ({merged['source'].value_counts().to_dict()})")

    print("\n4. Crop counts before filter:")
    counts = merged['label'].value_counts()
    for crop, n in counts.items():
        print(f"   {crop}: {n}")

    print(f"\n5. Filtering crops with < {MIN_SAMPLES_PER_CROP} samples...")
    crops_to_keep = counts[counts >= MIN_SAMPLES_PER_CROP].index
    merged = merged[merged['label'].isin(crops_to_keep)].reset_index(drop=True)
    print(f"   Crops kept: {sorted(merged['label'].unique())}")
    print(f"   Rows kept: {len(merged)}")

    print("\n6. Building crop→region map...")
    region_map = build_crop_region_map(
        zenodo_crops=set(zn['label'].unique()),
        kaggle_crops=set(kg['label'].unique())
    )
    # Filter to only kept crops
    region_map = {c: r for c, r in region_map.items() if c in crops_to_keep}
    for crop, regions in sorted(region_map.items()):
        print(f"   {crop}: {regions}")
    with open(BASE_DIR / 'crop_region_map.json', 'w') as f:
        json.dump(region_map, f, indent=2)

    print("\n7. Building validation ranges...")
    val_ranges = build_val_ranges(merged)
    for feat, r in val_ranges.items():
        print(f"   {feat}: [{r['min']:.1f}, {r['max']:.1f}]  "
              f"(recommended: [{r['q01']:.1f}, {r['q99']:.1f}])")
    with open(BASE_DIR / 'val_ranges.json', 'w') as f:
        json.dump(val_ranges, f, indent=2)

    print("\n8. Building crop_ranges.json...")
    crop_ranges = build_crop_ranges(merged)
    with open(BASE_DIR / 'crop_ranges.json', 'w') as f:
        json.dump(crop_ranges, f, indent=2)
    print(f"   → Saved crop_ranges.json ({len(crop_ranges)} crops)")

    print("\n9. Saving merged dataset...")
    merged_out = merged.drop(columns=['source'])
    merged_path = OUTPUT_DIR / 'Crop_recommendation_merged.csv'
    merged_out.to_csv(merged_path, index=False)
    print(f"   → Saved {merged_path} ({len(merged_out)} rows)")

    print("\n10. Label encoding summary...")
    labels = sorted(merged_out['label'].unique())
    le_map = {i: label for i, label in enumerate(labels)}
    print(f"    {len(labels)} crop classes")
    for i, label in le_map.items():
        count = len(merged_out[merged_out['label'] == label])
        print(f"      {i}: {label} ({count} samples)")

    print("\n✅ Data preparation complete!")
    return merged_out


if __name__ == "__main__":
    merged_df = main()
