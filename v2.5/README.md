# v2.5 — Continent-Aware Crop Recommendation System

> The latest generation: fuses global GAEZ suitability data with real African yield observations into a unified continent-aware XGBoost classifier.

## Screenshot

![Main UI](assets/screenshot_main.png)

## Features

- **33 crops** supported (maize, rice, cassava, yam, sorghum, banana, wheat, coffee, okra, citrus, sugarcane, sweet potato, groundnut, beans, soybean, tomatoes, orange, plantain, taro, tobacco, cocoa, cotton, millet, cowpea, cabbage, pepper, onion, ginger, pineapple, papaya, mango, tea, and more)
- **Continent-aware** — 5 continents: Africa, Asia, Global, North America, South America
- **9 features**: N, P, K, organic_carbon, temperature, humidity, pH, rainfall + continent_encoded
- **Top-5 predictions** with confidence bars
- **Local LLM explanations** via llama.cpp (optional GGUF model)
- **Interactive chat** for follow-up questions about crops, soil, and farming
- **Multi-language** — English and French
- **Weather integration** — fetch live weather for selected regions via Open-Meteo
- **Fertilizer advice** — per-crop NPK deficit/surplus recommendations
- **Auto-fill** — describe your field in natural language, LLM fills input values
- **Soil profiles** — auto-fill NPK/pH from predefined soil types
- **Region filtering** — filter suitable crops by geographic region

## Data Sources

| Source | Description | Rows | Crops |
|--------|-------------|------|-------|
| **GAEZ** (FAO) | Global Agro-Ecological Zones suitability rasters (50 crops) | 16,941 | 28 |
| **GROW-Africa** (Zenodo) | Real yield observations with GPS coordinates (LSMS + Point) | 3,714 | 24 |
| **WoSIS** (ISRIC) | Soil profiles for nearest-neighbor matching (27,081 profiles) | — | — |
| **Open-Meteo** | Free weather API for temperature/humidity | — | — |

### Dataset Pipeline

```
GAEZ rasters (.tif) ──> Sample points ──> WoSIS soil match ──> gov_dataset.csv
GROW-Africa (.xlsx) ──> GPS coords ──> WoSIS soil match ──> grow_dataset.csv
                                          │
                                          v
                                Unified Dataset (20,655 rows)
                                          │
                                          v
                              XGBoost Classifier (9 features)
                                          │
                                          v
                              xgboost_model.pkl + encoders
```

## Installation

```bash
cd v2.5
python -m venv venv
source venv/bin/activate
pip install customtkinter xgboost pandas numpy scikit-learn joblib
```

Optional (for LLM features):
```bash
pip install llama-cpp-python
```

## Usage

```bash
python main.py
```

### Workflow

1. **Enter values**: N, P, K, organic_carbon, temperature, humidity, pH, rainfall
2. **Select continent**: Africa, Asia, Global, North America, or South America
3. **Select region** (optional): filters suitable crops + enables weather fetch
4. **Select soil type** (optional): auto-fills NPK/pH from predefined profiles
5. **Click Predict**: model recommends top crop with confidence bars
6. **View explanation**: LLM-powered or rule-based explanation
7. **Ask follow-ups**: chat about farming tips, soil improvement, etc.
8. **Toggle language**: switch between English and French

### Auto-fill

Click "Describe field" and type something like:
```
I have clay soil in Nigeria, hot and humid with lots of rain, 
elevation around 300m, traditional farming methods.
```
The LLM extracts values and fills all input fields automatically.

## Model

- **Algorithm**: XGBoost Classifier
- **Features**: 9 (N, P, K, organic_carbon, temperature, humidity, pH, rainfall, continent_encoded)
- **Classes**: 33 crops
- **Validation accuracy**: 57.6%
- **Training data**: 20,655 rows (16,941 GAEZ + 3,714 GROW-Africa)

### Training

```bash
python scripts/merge_and_train.py
```

Or step by step:
```bash
python scripts/prepare_data.py
python scripts/train_model.py
```

## Project Structure

```
v2.5/
├── main.py                   Application entry point
├── install.py                Dependency installer
├── crop_ranges.json          Ideal ranges per crop
├── xgboost_model.pkl         Trained XGBoost model
├── label_encoder.pkl         Crop name encoder
├── continent_encoder.pkl     Continent label encoder
├── africa_crops.json         Africa-specific crop config
├── model_crops.json          All crop labels
├── assets/
│   └── screenshot_main.png
├── dataset/
│   ├── gov_dataset.csv       GAEZ-based dataset (28 crops)
│   └── grow_dataset.csv      GROW-Africa dataset (24 crops)
├── scripts/
│   ├── merge_and_train.py    Unified training pipeline
│   ├── prepare_data.py       Data preparation
│   ├── train_model.py        8-feature trainer
│   ├── train_yield_model.py  Yield regression (deprecated)
│   └── weather.py            Open-Meteo API client
└── README.md                 This file
```

## Crop Ranges

| Crop | N (kg/ha) | P (kg/ha) | K (kg/ha) | Temp (°C) | Humidity (%) | pH | Rain (mm) |
|------|-----------|-----------|-----------|-----------|-------------|-----|-----------|
| Banana | 20-50 | 10-30 | 20-60 | 24-32 | 65-90 | 5.5-7.5 | 100-250 |
| Cassava | 5-20 | 5-15 | 10-30 | 20-35 | 50-80 | 4.5-7.5 | 50-200 |
| Coffee | 15-40 | 10-25 | 15-45 | 18-28 | 60-85 | 5.0-6.5 | 80-200 |
| Maize | 60-120 | 20-60 | 20-60 | 18-32 | 50-80 | 5.5-7.5 | 50-200 |
| Rice | 50-100 | 15-40 | 15-40 | 20-35 | 70-90 | 5.0-7.5 | 100-250 |
| Sorghum | 30-80 | 10-30 | 15-40 | 20-35 | 40-70 | 5.5-8.0 | 30-150 |
| Soybean | 10-30 | 10-30 | 10-30 | 18-30 | 50-80 | 6.0-7.5 | 50-150 |
| Yam | 20-50 | 10-25 | 20-50 | 24-32 | 65-90 | 5.0-7.0 | 100-250 |

## License

MIT License
