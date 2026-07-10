# v0.1 — Crop Recommendation System GUI (Tkinter + Random Forest)

> First generation: a simple interactive desktop app using Random Forest Classifier and Tkinter.

## Features

- **Random Forest classifier** (100 trees, max_depth=15)
- **22 crops** supported
- **7 features**: N, P, K, temperature, humidity, pH, rainfall
- **Simple Tkinter GUI** with input validation
- **Confidence scores** per prediction

## Installation

```bash
cd v0.1
pip install pandas numpy scikit-learn
```

## Usage

```bash
python gui.py
```

## Dataset

Trained on the Kaggle Crop Recommendation Dataset (Atharva Ingle, 2020):
- 22 crops
- 7 features
- 2,200 samples

## License

Apache 2.0
