# 🌾 Crop Recommendation System GUI

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

An interactive desktop application that recommends the most suitable crop to grow based on soil nutrient levels and environmental conditions. This project uses a **Random Forest Classifier** trained on public agricultural data from Kaggle.

> **Note:** This is an **example/demonstration project** built for educational purposes. It showcases a complete Machine Learning pipeline integrated with a user-friendly GUI.

---

# 📊 Dataset Source

The model is trained on the **[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)** by **Atharva Ingle** (published on Kaggle, 2020).

- **Context:** The dataset was created by augmenting rainfall, climate, and fertilizer data available for India.
- **License:** Apache 2.0
- **Goal:** Enable users to build predictive models for recommending crops based on farm parameters.

## Data Fields Used

| Feature | Description | Range in Dataset |
|----------|-------------|------------------|
| N | Nitrogen ratio in soil | 0–140 |
| P | Phosphorous ratio in soil | 0–100 |
| K | Potassium ratio in soil | 0–200 |
| temperature | Temperature in °C | 8–45 |
| humidity | Relative humidity in % | 14–99 |
| ph | Soil pH value | 3.5–9.9 |
| rainfall | Rainfall in mm | 20–298 |

---

# 🎯 Project Purpose

- Demonstrate a complete ML pipeline (training → inference)
- Provide a user-friendly GUI for model interaction
- Serve as a learning example for integrating scikit-learn with Tkinter
- Show proper dataset attribution and documentation

---

# ⚙️ How It Works

1. User inputs 7 soil & climate parameters
2. GUI validates inputs against realistic ranges
3. Random Forest model predicts the optimal crop
4. System displays predicted crop + confidence score

---

# 🚀 How to Run

## What You Need (Prerequisites)

- **Python 3.7 or higher** installed on your system
- **pip** (Python package manager, usually included with Python)
- **Internet connection** (only required the first time to install dependencies)

---

## Step-by-Step Instructions

### 1. Clone or Download This Repository

```bash
git clone https://github.com/yourusername/crop-recommendation-gui.git
cd crop-recommendation-gui
```

> If you do not use Git, simply download the ZIP file and extract it.

---

### 2. Install Required Python Packages

Open a terminal or command prompt inside the project folder and run:

```bash
pip install pandas numpy scikit-learn
```

> These are the **only external modules** needed.  
> Standard library modules such as `tkinter` and `warnings` are already included with Python.

---

### 3. Place the Dataset

1. Download `Crop_recommendation.csv` from Kaggle:
   - https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

2. Create a folder named `dataset` inside the project root.

3. Move the CSV file into the `dataset/` folder.

4. Ensure the filename is exactly:

```text
Crop_recommendation.csv
```

Your folder structure should look like this:

```text
crop-recommendation-gui/
├── gui.py
├── LICENSE
├── dataset/
│   └── Crop_recommendation.csv
└── README.md
```

---

### 4. Run the Application

```bash
python gui.py
```

The GUI window will open. Enter values and click **Predict Crop**.

---

# 🔁 Using Your Own Dataset

You can replace the provided dataset with your own **as long as it follows the exact same format**.

---

## Requirements for a Custom Dataset

Your CSV file must include:

### Required Columns (Case-Sensitive)

```text
N
P
K
temperature
humidity
ph
rainfall
label
```

### Additional Requirements

- CSV format (comma-separated values)
- Same column names
- Label column contains crop names (`rice`, `wheat`, `maize`, etc.)
- Same structure as the original dataset

> The column order is not critical because the code reads by column name, but keeping the same order is recommended.

---

## How to Replace the Dataset

1. Prepare your CSV file with the required columns.
2. Replace the existing file located at:

```text
dataset/Crop_recommendation.csv
```

3. Keep the exact same filename:

```text
Crop_recommendation.csv
```

4. Run the application again:

```bash
python gui.py
```

The model will automatically retrain using your new dataset.

---

## Important Note

The GUI currently validates input ranges using hardcoded values based on the original dataset.

Example:

```python
"N": (0, 140)
```

If your dataset uses different realistic ranges, you may need to modify the `ranges` dictionary inside `gui.py`.

---

# 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Parameters | `n_estimators=100`, `max_depth=15`, `random_state=42`, `n_jobs=-1` |
| Train/Test Split | 80/20 with stratification |
| Input Features | 7 |
| Output | Crop label prediction |

### Input Features

- N
- P
- K
- temperature
- humidity
- ph
- rainfall

---

# 📸 Screenshot

<img width="500" height="684" alt="Screenshot from 2026-05-12 01-19-00" src="https://github.com/user-attachments/assets/8a5140c9-b5a5-4709-8154-87802af3a2a6" />


---

# 🙏 Acknowledgements

- **Dataset Author:** [Atharva Ingle](https://www.kaggle.com/atharvaingle)
- **Dataset:** [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Dataset License:** Apache 2.0



⭐ If this project helps you learn or build something useful, please give proper credit to the original dataset author and repository.
