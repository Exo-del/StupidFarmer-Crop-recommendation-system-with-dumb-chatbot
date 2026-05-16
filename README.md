# 🌾 Crop Recommendation System GUI

A modern AI-powered desktop application that analyzes soil and climate conditions to recommend the most suitable crops.  
The system combines machine learning with a local Large Language Model (LLM) to provide both accurate crop predictions and human-readable explanations.

---

# ✨ Features

- **🧠 AI Crop Prediction**
  Uses a trained XGBoost classifier to recommend crops based on:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - pH
  - Rainfall

- **🤖 Local LLM Explanations**
  Supports local GGUF models through `llama.cpp` to explain predictions in natural language.

- **💬 Interactive Chat**
  Ask follow-up questions about:
  - Recommended crops
  - Soil conditions
  - Farming tips
  - Environmental suitability

- **🎨 Modern Desktop UI**
  Built with `customtkinter` using a clean dark-themed interface.

- **⚡ Fallback Explanation System**
  If the LLM is unavailable, the app automatically switches to a built-in rule-based explanation engine.

---

# 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| GUI | CustomTkinter |
| ML Framework | scikit-learn |
| Model | XGBoost |
| Data Processing | pandas, numpy |
| Local AI | llama-cpp-python |

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/CRS_V2.git
cd CRS_V2
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Linux/macOS

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
python install.py
```

> **Note:**  
> If `llama-cpp-python` fails to install because of missing C++ build tools, the application will still work using the fallback explanation system.

---

# 📁 Project Structure

```text
CRS_V2/
│
├── LLM/
│   └── *.gguf                         # Local LLM models
│
├── dataset/
│   └── Crop_recommendation.csv       # Training dataset
│
├── __pycache__/
│
├── crop_ranges.json                  # Crop condition ranges
├── data-analysis.ipynb               # Dataset analysis notebook
├── install.py                        # Dependency installer
├── main.py                           # Main application
├── xgboost_model.pkl                 # Trained ML model
│
└── README.md
```

---

# 🤖 Local LLM Setup

Place any supported GGUF model inside the `LLM/` folder.

Example:

```text
LLM/
└── LittleLamb-ToolCalling.f16.gguf
```

Compatible with:
- TinyLlama
- Phi
- Gemma
- LittleLamb
- Other GGUF models supported by `llama.cpp`

---

# 📊 Dataset

The dataset is stored inside:

```text
dataset/Crop_recommendation.csv
```

It contains agricultural parameters used for training and prediction.

### Features Used

| Feature | Description |
|---|---|
| N | Nitrogen level |
| P | Phosphorus level |
| K | Potassium level |
| temperature | Temperature in °C |
| humidity | Relative humidity |
| ph | Soil pH |
| rainfall | Rainfall in mm |

---

# 🎮 Usage

Run the application:

```bash
python main.py
```

### Workflow

1. Enter soil and climate values.
2. Click **Predict Crop**.
3. View:
   - Predicted crop
   - Confidence score
   - AI explanation
4. Use the chat section for follow-up questions.

---

# 📈 Model

The application uses a pre-trained:

- **XGBoost Classifier**

Stored as:

```text
xgboost_model.pkl
```

---

# 🧪 Data Analysis

Dataset exploration and analysis are available in:

```text
data-analysis.ipynb
```

This notebook may include:
- Crop distribution
- Outlier detection
- Correlation analysis
- Feature statistics
- Visualization experiments
---
---

# 📸 Screenshots

## Main Interface

<p align="center">
  <img src="screenshots/main-ui.png" width="900" alt="Main Application UI">
</p>

---

## Prediction Result

<img width="950" height="918" alt="image" src="https://github.com/user-attachments/assets/4b26514c-dd5e-441a-abca-cc0425104598" />



---

## Interactive AI Chat

<img width="950" height="918" alt="image" src="https://github.com/user-attachments/assets/1adf1bda-e7dd-48c8-8550-81696074a661" />


---

# 🤝 Contributing

Contributions are welcome.

You can help improve:
- UI/UX
- Model performance
- Dataset quality
- LLM integration
- Prediction explanations

---

# 📄 License

This project is open-source and available under the MIT License.
