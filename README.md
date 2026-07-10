# 🌾 Crop Recommendation System

Three generations of crop recommendation tools in one monorepo.

---

## 📦 Versions

| Version | Description | Stack | Directory |
|---------|-------------|-------|-----------|
| **v0.1** | Simple desktop app with Random Forest + Tkinter | `scikit-learn`, `Tkinter` | [`v0.1/`](./v0.1) |
| **v1.0** | Modern UI with XGBoost + local LLM explanations | `XGBoost`, `CustomTkinter`, `llama.cpp` | [`v1.0/`](./v1.0) |
| **v2.5** | Continent-aware prediction with GAEZ + GROW-Africa fusion | `XGBoost`, `GAEZ`, `WoSIS`, `Open-Meteo` | [`v2.5/`](./v2.5) |

## 🏷️ Releases

- [**v2.5**](https://github.com/Exo-del/StupidFarmer-Crop-recommendation-system-with-dumb-chatbot/releases/tag/v2.5) — Latest: continent-aware, 33 crops, 9 features
- [**v1.0**](https://github.com/Exo-del/StupidFarmer-Crop-recommendation-system-with-dumb-chatbot/releases/tag/v1.0) — XGBoost + LLM chatbot
- [**v0.1**](https://github.com/Exo-del/StupidFarmer-Crop-recommendation-system-with-dumb-chatbot/releases/tag/v0.1) — Original Random Forest + Tkinter

## 🚀 Quick Start

```bash
# Latest version
cd v2.5
python -m venv venv
source venv/bin/activate
python install.py
python main.py
```

## 📁 Structure

```
├── v0.1/          # Tkinter + Random Forest (7 features, 22 crops)
│   ├── gui.py
│   └── dataset/
├── v1.0/          # CustomTkinter + XGBoost + LLM (7 features, 22 crops)
│   ├── main.py
│   └── dataset/
├── v2.5/          # Continent-aware XGBoost (9 features, 33 crops)
│   ├── main.py
│   ├── scripts/   # Training pipelines
│   └── dataset/
├── LLM/           # Shared GGUF model directory
└── README.md
```

## 📜 License

MIT License
