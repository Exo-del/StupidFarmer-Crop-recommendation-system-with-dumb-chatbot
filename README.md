# Crop Recommendation System

An AI-powered desktop application that analyzes soil nutrients, climate conditions, and geographic location to recommend the most suitable crops. Spans three generations of ML tooling.

## Versions

| Version | Stack | Features | Directory |
|---------|-------|----------|-----------|
| **v2.5** | XGBoost, CustomTkinter, GAEZ + GROW-Africa | 33 crops, continent-aware (5 continents), 9 features, top-5 confidence bars, LLM explanations, multi-language | [`v2.5/`](./v2.5) |
| **v1.0** | XGBoost, CustomTkinter, llama.cpp | 22 crops, 7 features, LLM chat, fallback explanations | [`v1.0/`](./v1.0) |
| **v0.1** | scikit-learn Random Forest, Tkinter | 22 crops, 7 features, basic GUI | [`v0.1/`](./v0.1) |

## Quick Start (latest)

```bash
cd v2.5
python -m venv venv
source venv/bin/activate
python install.py
python main.py
```

## Releases

| Release | Link |
|---------|------|
| v2.5 — Continent-Aware | https://github.com/Exo-del/StupidFarmer-Crop-recommendation-system-with-dumb-chatbot/releases/tag/v2.5 |
| v1.0 — XGBoost + LLM | https://github.com/Exo-del/StupidFarmer-Crop-recommendation-system-with-dumb-chatbot/releases/tag/v1.0 |
| v0.1 — Random Forest + Tkinter | https://github.com/Exo-del/StupidFarmer-Crop-recommendation-system-with-dumb-chatbot/releases/tag/v0.1 |

## Repository Structure

```
.
├── v0.1/          Tkinter + Random Forest
│   ├── gui.py     Standalone desktop app
│   └── dataset/   Crop_recommendation.csv
├── v1.0/          CustomTkinter + XGBoost + LLM
│   ├── main.py    Main application
│   ├── install.py Dependency installer
│   └── dataset/   Training data
├── v2.5/          Continent-aware XGBoost (active version)
│   ├── main.py    Main application with continent dropdown
│   ├── scripts/   Training pipelines
│   ├── assets/    Screenshots
│   └── dataset/   GAEZ + GROW-Africa data
├── LLM/           Shared GGUF model directory (gitignored)
└── README.md      This file
```

## License

MIT License
