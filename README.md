# 🌾 Crop Recommendation System

A monorepo spanning 3 generations of crop recommendation tools:

| Version | Description |
|---------|-------------|
| **v0.1** | `gui.py` — Original Tkinter + Random Forest classifier (7 features) |
| **v1.0** | `main.py` — CustomTkinter + XGBoost + local LLM explanations |
| **v2.5** | Continent-aware XGBoost with GAEZ + GROW-Africa fusion, 33 crops, multi-crop visualization |

---

## ✨ Features (v2.5)

- **🧠 AI Crop Prediction** — XGBoost classifier trained on 9 features (N, P, K, organic_carbon, temperature, humidity, ph, rainfall + continent_encoded)
- **🌍 Continent-Aware** — Dropdown selects Africa, Asia, Global, North America, or South America
- **📊 Multi-Crop Output** — Top-5 recommendations with confidence bars
- **🤖 Local LLM Explanations** — GGUF model via llama.cpp for natural-language reasoning
- **💬 Interactive Chat** — Ask follow-ups about crops, soil, farming tips
- **🎨 Modern UI** — CustomTkinter dark theme
- **⚡ Fallback Explanations** — Rule-based engine when LLM is unavailable

---

## 🚀 Quick Start

```bash
python -m venv venv
source venv/bin/activate
python install.py
python main.py        # v1.0/v2.5
# or
python gui.py         # v0.1 legacy
```

---

## 📁 Structure

```
├── main.py                  # v2.5 continent-aware app
├── gui.py                   # v0.1 legacy Tkinter app
├── install.py               # Dependency installer
├── crop_ranges.json         # Crop condition ranges
├── xgboost_model.pkl        # Trained model
├── continent_encoder.pkl    # Continent label encoder
├── label_encoder.pkl        # Crop label encoder
├── dataset/
│   ├── gov_dataset.csv      # GAEZ-based dataset (28 crops)
│   └── grow_dataset.csv     # GROW-Africa dataset (24 crops)
├── LLM/
│   └── *.gguf               # Local LLM models
└── scripts/
    ├── merge_and_train.py   # Unified training pipeline
    └── train_model.py       # 8-feature trainer
```

---

## 📜 License

MIT License
