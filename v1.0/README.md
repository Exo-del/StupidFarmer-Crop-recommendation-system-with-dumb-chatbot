# v1.0 — Crop Recommendation System (CustomTkinter + XGBoost + LLM)

> Second generation: modern dark-themed UI with XGBoost and optional local LLM explanations via llama.cpp.

## Features

- **XGBoost classifier** trained on 7 soil/climate features (N, P, K, temperature, humidity, pH, rainfall)
- **22 crops** supported
- **Local LLM integration** via llama.cpp GGUF models
- **Interactive chat** for follow-up questions
- **Fallback rule-based explanations** when LLM is unavailable
- **Dark-themed modern UI** built with CustomTkinter

## Installation

```bash
cd v1.0
python -m venv venv
source venv/bin/activate
pip install customtkinter xgboost pandas numpy scikit-learn joblib
```

Optional (for LLM):
```bash
pip install llama-cpp-python
```

## Usage

```bash
python main.py
```

### LLM Setup

Place a GGUF model file in `LLM/` directory:
```
LLM/
└── your-model.gguf
```

## Dataset

Trained on the Kaggle Crop Recommendation Dataset (Atharva Ingle, 2020):
- 22 crops
- 7 features
- 2,200 samples

## License

MIT License
