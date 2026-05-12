# 🌾 Crop Recommendation System GUI

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

An interactive desktop application that recommends the most suitable crop to grow based on soil nutrient levels and environmental conditions. This project uses a **Random Forest Classifier** trained on public agricultural data from Kaggle.

> **Note:** This is an **example/demonstration project** built for educational purposes. It showcases a complete Machine Learning pipeline integrated with a user-friendly GUI.

## 📊 Dataset Source

The model is trained on the **[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)** by **Atharva Ingle** (published on Kaggle, 2020).

- **Context:** The dataset was created by augmenting rainfall, climate, and fertilizer data available for India.
- **License:** Apache 2.0
- **Goal:** Enable users to build predictive models for recommending crops based on farm parameters.

### Data Fields Used

| Feature | Description | Range in Dataset |
|---------|-------------|------------------|
| N | Nitrogen ratio in soil | 0–140 |
| P | Phosphorous ratio in soil | 0–100 |
| K | Potassium ratio in soil | 0–200 |
| temperature | Temperature in °C | 8–45 |
| humidity | Relative humidity in % | 14–99 |
| ph | Soil pH value | 3.5–9.9 |
| rainfall | Rainfall in mm | 20–298 |

## 🎯 Project Purpose

- Demonstrate a complete ML pipeline (training → persistence → inference)
- Provide a user-friendly GUI for model interaction
- Serve as a learning example for integrating scikit-learn with Tkinter
- Show proper dataset attribution and documentation

## ⚙️ How It Works

1. User inputs 7 soil & climate parameters
2. GUI validates inputs against realistic ranges
3. Random Forest model predicts the optimal crop
4. System displays predicted crop + confidence score

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn
