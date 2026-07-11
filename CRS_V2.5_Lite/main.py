import json
import os
import re
import sys
import threading
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox
import warnings
warnings.filterwarnings('ignore')

try:
    import customtkinter as ctk
except ImportError:
    raise ImportError("customtkinter required: pip install customtkinter")

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# ============ CONFIG ============
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
FIELD_LABELS = [
    ('Nitrogen (N)', 'N', '0 – 250 kg/ha'),
    ('Phosphorus (P)', 'P', '0 – 100 kg/ha'),
    ('Potassium (K)', 'K', '0 – 400 kg/ha'),
    ('Temperature', 'temperature', '5 – 40 °C'),
    ('Humidity', 'humidity', '10 – 100 %'),
    ('pH Level', 'ph', '4.0 – 9.0'),
    ('Rainfall', 'rainfall', '0 – 4000 mm'),
]

INPUT_RANGES = {
    'N': (0, 250), 'P': (0, 100), 'K': (0, 400),
    'temperature': (5, 40), 'humidity': (10, 100),
    'ph': (4.0, 9.0), 'rainfall': (0, 4000)
}

# --- Soil type profiles (median NPK/pH from real data) ---
SOIL_PROFILES = {
    'sandy':        {'N': 25, 'P': 15, 'K': 20, 'ph': 6.0, 'name': 'Sandy (low CEC, drains fast)'},
    'loam':         {'N': 45, 'P': 30, 'K': 35, 'ph': 6.5, 'name': 'Loam (balanced, ideal)'},
    'clay':         {'N': 60, 'P': 40, 'K': 50, 'ph': 6.8, 'name': 'Clay (high CEC, holds water)'},
    'silt':         {'N': 50, 'P': 35, 'K': 40, 'ph': 6.6, 'name': 'Silt (fertile, moderate drainage)'},
    'peat':         {'N': 80, 'P': 20, 'K': 30, 'ph': 5.5, 'name': 'Peat (high organic, acidic)'},
    'chalky':       {'N': 30, 'P': 25, 'K': 30, 'ph': 7.8, 'name': 'Chalky (alkaline, free-draining)'},
    'laterite':     {'N': 20, 'P': 10, 'K': 15, 'ph': 5.2, 'name': 'Laterite (tropical, Fe/Al oxides)'},
    'volcanic':     {'N': 70, 'P': 45, 'K': 60, 'ph': 6.3, 'name': 'Volcanic (rich, young soils)'},
}

# --- Climate zone presets (temperature, humidity, rainfall) ---
CLIMATE_PRESETS = {
    'tropical':        {'temperature': 27, 'humidity': 80, 'rainfall': 2000, 'name': 'Tropical (hot, wet year-round)'},
    'subtropical':     {'temperature': 23, 'humidity': 70, 'rainfall': 1200, 'name': 'Subtropical (hot summers, mild winters)'},
    'mediterranean':   {'temperature': 18, 'humidity': 60, 'rainfall': 600, 'name': 'Mediterranean (dry summers, wet winters)'},
    'temperate':       {'temperature': 14, 'humidity': 65, 'rainfall': 800, 'name': 'Temperate (4 seasons, moderate)'},
    'semi_arid':       {'temperature': 25, 'humidity': 40, 'rainfall': 400, 'name': 'Semi-arid (hot, low rainfall)'},
    'arid':            {'temperature': 28, 'humidity': 25, 'rainfall': 150, 'name': 'Arid (desert, very low rainfall)'},
    'highland':        {'temperature': 12, 'humidity': 60, 'rainfall': 1000, 'name': 'Highland/Cool (elevation-driven)'},
    'continental':     {'temperature': 10, 'humidity': 55, 'rainfall': 500, 'name': 'Continental (hot summers, cold winters)'},
}

# ============ AUTO-SETUP ============
def run_first_time_setup():
    """Train model if missing."""
    model_path = BASE_DIR / 'xgboost_model.pkl'
    if model_path.exists():
        return True

    print("First run: training model...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-c', '''
import pandas as pd, numpy as np, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "dataset" / "Crop_recommendation.csv")
df["label"] = df["label"].str.strip().str.lower()
FEATS = ["N","P","K","temperature","humidity","ph","rainfall"]
X = df[FEATS].values
le = LabelEncoder()
y = le.fit_transform(df["label"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_lambda=1.0, reg_alpha=0.5,
    random_state=42, eval_metric="mlogloss", verbosity=0)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
joblib.dump(model, BASE / "xgboost_model.pkl")
joblib.dump(le, BASE / "label_encoder.pkl")
crop_ranges = {}
for crop in sorted(df["label"].unique()):
    sub = df[df["label"] == crop]
    crop_ranges[crop] = {
        "N": [int(sub["N"].min()), int(sub["N"].max())],
        "P": [int(sub["P"].min()), int(sub["P"].max())],
        "K": [int(sub["K"].min()), int(sub["K"].max())],
        "temperature": [round(float(sub["temperature"].min()),1), round(float(sub["temperature"].max()),1)],
        "humidity": [round(float(sub["humidity"].min()),1), round(float(sub["humidity"].max()),1)],
        "ph": [round(float(sub["ph"].min()),2), round(float(sub["ph"].max()),2)],
        "rainfall": [round(float(sub["rainfall"].min()),1), round(float(sub["rainfall"].max()),1)],
    }
with open(BASE / "crop_ranges.json", "w") as f:
    json.dump(crop_ranges, f, indent=2)
print("Setup complete")
'''],
            capture_output=True, text=True, cwd=BASE_DIR
        )
        if result.returncode != 0:
            print("Setup failed:", result.stderr)
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Setup error: {e}")
        return False


# ============ LOAD ARTIFACTS ============
def load_model():
    if not run_first_time_setup():
        return None, None, 0.0

    model = joblib.load(BASE_DIR / 'xgboost_model.pkl')
    le = joblib.load(BASE_DIR / 'label_encoder.pkl')

    # Compute accuracy on holdout
    df = pd.read_csv(BASE_DIR / 'dataset' / 'Crop_recommendation.csv')
    df['label'] = df['label'].str.strip().str.lower()
    X = df[FEATURES].values
    y = le.transform(df['label'])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    acc = model.score(X_test, y_test)
    return model, le, acc


def load_llm():
    llm_dir = BASE_DIR / 'LLM'
    models = sorted(llm_dir.glob('*.gguf'))
    if not models or Llama is None:
        return None
    try:
        return Llama(model_path=str(models[0]), n_threads=min(4, os.cpu_count() or 1), n_ctx=8000)
    except Exception:
        return None


def load_crop_ranges():
    path = BASE_DIR / 'crop_ranges.json'
    if not path.exists():
        return {}
    with open(path) as f:
        return {k.lower(): v for k, v in json.load(f).items()}


def build_crop_summary(ranges):
    if not ranges:
        return "No crop range data."
    lines = []
    for crop in sorted(ranges.keys()):
        v = ranges[crop]
        lines.append(
            f"{crop}: N {v['N'][0]}-{v['N'][1]}, P {v['P'][0]}-{v['P'][1]}, "
            f"K {v['K'][0]}-{v['K'][1]}, T {v['temperature'][0]}-{v['temperature'][1]}, "
            f"H {v['humidity'][0]}-{v['humidity'][1]}, pH {v['ph'][0]}-{v['ph'][1]}, "
            f"R {v['rainfall'][0]}-{v['rainfall'][1]}"
        )
    return "\n".join(lines)


# ============ RULE-BASED EXPLANATION ============
ENV_FACTS = {
    'rice': 'thrives in warm, flooded fields with steady water',
    'maize': 'prefers warm, sunny weather and well-drained soil',
    'beans': 'benefits from warm soil and moderate rainfall',
    'cassava': 'tolerates poor soils and dry conditions once established',
    'guinea_corn': 'thrives in warm, well-drained soils with moderate rain',
    'orange': 'prefers warm, subtropical weather and well-drained soil',
    'pepper': 'grows best in warm, humid conditions with well-drained soil',
    'soybean': 'does well in warm, well-drained soils with consistent moisture',
    'tomatoes': 'prefers warm, sunny conditions with well-drained, fertile soil',
    'yam': 'thrives in warm, humid conditions with loose, well-drained soil',
    'apple': 'grows best in cooler climates with well-drained, fertile soil',
    'banana': 'loves warm, humid conditions with plenty of water and rich soil',
    'blackgram': 'favors warm, humid weather and well-drained soils',
    'chickpea': 'does well in drier, well-drained fields with moderate moisture',
    'coconut': 'flourishes in tropical, coastal conditions with ample moisture',
    'coffee': 'favors warm, shaded areas with rich, well-drained soil',
    'cotton': 'prefers warm temperatures and well-drained, fertile soils',
    'grapes': 'prefer sunny, dry weather and well-drained soil for healthy vines',
    'jute': 'grows best in warm, humid regions with plenty of moisture',
    'kidneybeans': 'grows best in warm soil and needs consistent moisture during flowering',
    'lentil': 'prefers cool, dry conditions and well-drained sandy soil',
    'mango': 'flourishes in warm, frost-free climates with good drainage',
    'mothbeans': 'suited to hot, semi-arid conditions and low rainfall',
    'mungbean': 'thrives in warm soils with moderate humidity and good drainage',
    'muskmelon': 'does well in warm, sunny fields with good drainage',
    'papaya': 'thrives in warm, humid conditions with rich, well-drained soil',
    'pigeonpeas': 'likes warm climates and can tolerate dry spells once established',
    'pomegranate': 'grows well in warm, dry climates with deep, well-drained soil',
    'watermelon': 'likes hot weather, plenty of sun, and consistent moisture',
}

def rule_explanation(crop, vals):
    cr = load_crop_ranges().get(crop.lower(), {})
    keys = FEATURES
    labels = {'N': 'nitrogen', 'P': 'phosphorus', 'K': 'potassium',
              'temperature': 'temperature', 'humidity': 'humidity',
              'ph': 'pH', 'rainfall': 'rainfall'}
    matched = []
    for k, v in zip(keys, vals):
        rng = cr.get(k)
        if not rng or len(rng) != 2:
            continue
        try:
            lo, hi = float(rng[0]), float(rng[1])
            if lo > hi or (lo == 0 and hi == 0):
                continue
            if lo <= v <= hi:
                matched.append((k, lo, hi))
        except Exception:
            continue
    fact = ENV_FACTS.get(crop.lower(), 'is suited to these conditions')
    if len(matched) == len(keys):
        return f"Did you know {crop} is a strong match? All your values fall within its ideal ranges, and {fact}."
    elif matched:
        names = [labels[k] for k, _, _ in matched[:3]]
        return f"Did you know {crop} is recommended because {', '.join(names)} are within ideal range, and {fact}."
    else:
        return f"Did you know {crop} is recommended for these soil and climate conditions, and {fact}."


# ============ WEATHER API ============
def fetch_weather(city):
    """Open-Meteo geocoding + forecast."""
    try:
        # Geocode
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10
        ).json()
        if not geo.get('results'):
            return None
        loc = geo['results'][0]
        lat, lon = loc['latitude'], loc['longitude']
        name = f"{loc['name']}, {loc.get('country_code', '')}"

        # Current weather
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current_weather": "true",
                "hourly": "relativehumidity_2m,precipitation",
                "timezone": "auto"
            },
            timeout=10
        ).json()

        curr = wx.get('current_weather', {})
        hum = wx.get('hourly', {}).get('relativehumidity_2m', [None])[0]
        prec = wx.get('hourly', {}).get('precipitation', [None])[0]

        return {
            'city': name,
            'temperature': curr.get('temperature'),
            'humidity': hum,
            'rainfall': prec,
        }
    except Exception:
        return None


# ============ GUI ============
class CropApp:
    def __init__(self, root, model, le, accuracy, llm=None):
        self.root = root
        self.model = model
        self.le = le
        self.accuracy = accuracy
        self.llm = llm
        self.crop_ranges = load_crop_ranges()
        self.crop_summary = build_crop_summary(self.crop_ranges)
        self.prediction_ctx = None
        self.chat_history = []

        self.root.title("🌱 SeedBrain")
        self.root.geometry("950x850")
        self.root.minsize(850, 750)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.main = ctk.CTkScrollableFrame(self.root, fg_color=("#121212", "#1a1a1a"), border_width=0)
        self.main.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.main.grid_columnconfigure((0, 1), weight=1, uniform="col")
        self.main.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_left_panel()
        self._build_right_panel()

    def _build_header(self):
        hero = ctk.CTkFrame(self.main, fg_color=("#19232d", "#1e2d39"),
                            corner_radius=22, border_width=1, border_color=("#34596f", "#1c3446"))
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        hero.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(hero, text="🌱 SeedBrain — Crop Recommendation",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color=("#d7ffd9", "#d7ffd9")).grid(row=0, column=0, sticky="w", padx=24, pady=(20, 6))

        ctk.CTkLabel(hero,
            text="Real-data model (FUBC + WorldClim) • 29 crops • Soil & climate presets • Weather API",
            font=ctk.CTkFont(family="Segoe UI", size=11), text_color="gray70",
            wraplength=900, justify="left").grid(row=1, column=0, sticky="w", padx=24, pady=(0, 16))

        status = ctk.CTkFrame(hero, fg_color="transparent")
        status.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 16))
        status.grid_columnconfigure(0, weight=1)

        acc_badge = ctk.CTkFrame(status, fg_color=("#203d2f", "#1e3730"),
                                  corner_radius=18, border_width=1, border_color=("#2e6d4b", "#1b3f31"))
        acc_badge.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(acc_badge, text=f"🎯 Accuracy: {self.accuracy*100:.1f}%",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#98ffb5"
        ).pack(padx=18, pady=12)

        llm_badge = ctk.CTkFrame(status, fg_color=("#22344f", "#1f2d43"),
                                  corner_radius=18, border_width=1, border_color=("#3a5a9e", "#1b2a3f"))
        llm_badge.grid(row=0, column=1, sticky="e")
        self.llm_lbl = ctk.CTkLabel(llm_badge,
            text=f"LLM: {'✅ loaded' if self.llm else '❌ unavailable'}",
            font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#b9d2ff")
        self.llm_lbl.pack(padx=18, pady=12)

    def _build_left_panel(self):
        left = ctk.CTkFrame(self.main, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        left.grid_columnconfigure(0, weight=1)

        # --- Input Card ---
        card = ctk.CTkFrame(left, fg_color=("#171f26", "#1e2931"),
                             corner_radius=20, border_width=1, border_color=("#2f546a", "#1d3345"))
        card.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        card.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(card, text="🧪 Soil & Climate Inputs",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(18, 10))

        # Soil type selector
        soil_frame = ctk.CTkFrame(card, fg_color="transparent")
        soil_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=6)
        soil_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(soil_frame, text="Soil Type",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")).grid(row=0, column=0, sticky="w", padx=(0, 10))

        soil_options = ["-- Manual --"] + [f"{v['name']}" for v in SOIL_PROFILES.values()]
        self.soil_var = ctk.StringVar(value=soil_options[0])
        ctk.CTkOptionMenu(soil_frame, variable=self.soil_var, values=soil_options,
            command=self._on_soil_change, height=36, corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        ).grid(row=0, column=1, sticky="ew", padx=(0, 10))

        # Climate zone selector
        ctk.CTkLabel(soil_frame, text="Climate Zone",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(8, 0))

        climate_options = ["-- Manual --"] + [f"{v['name']}" for v in CLIMATE_PRESETS.values()]
        self.climate_var = ctk.StringVar(value=climate_options[0])
        ctk.CTkOptionMenu(soil_frame, variable=self.climate_var, values=climate_options,
            command=self._on_climate_change, height=36, corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        ).grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(8, 0))

        # Weather fetch
        ctk.CTkLabel(soil_frame, text="Weather",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")).grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(8, 0))

        loc_frame = ctk.CTkFrame(soil_frame, fg_color="transparent")
        loc_frame.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(8, 0))
        loc_frame.grid_columnconfigure(0, weight=1)

        self.city_entry = ctk.CTkEntry(loc_frame, placeholder_text="City name (e.g., Casablanca)",
            height=36, corner_radius=10, font=ctk.CTkFont(family="Segoe UI", size=11))
        self.city_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.city_entry.bind('<Return>', lambda e: self._fetch_weather())

        ctk.CTkButton(loc_frame, text="☁ Fetch", width=80, height=36, corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            command=self._fetch_weather
        ).grid(row=0, column=1)

        # Field entries
        self.entries = {}
        for i, (label, key, hint) in enumerate(FIELD_LABELS):
            row = (i // 2) + 2
            col = i % 2
            frame = ctk.CTkFrame(card, fg_color="transparent")
            frame.grid(row=row, column=col, sticky="nsew", padx=10, pady=8)
            frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(frame, text=label,
                font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), anchor="w"
            ).grid(row=0, column=0, sticky="w")
            entry = ctk.CTkEntry(frame, placeholder_text=hint, height=38,
                corner_radius=10, border_width=2, font=ctk.CTkFont(family="Segoe UI", size=11))
            entry.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            self.entries[key] = entry

        # Action buttons
        act = ctk.CTkFrame(left, fg_color="transparent")
        act.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        act.grid_columnconfigure(0, weight=1)

        self.predict_btn = ctk.CTkButton(act, text="🌾 Predict Crop", height=44,
            corner_radius=14, font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=("#1f7a4f", "#159e5a"), hover_color=("#1f8b5d", "#1faa6d"),
            command=self._predict)
        self.predict_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(act, text="Clear", height=44, width=90, corner_radius=14,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=("#8b1f2d", "#a5283d"), hover_color=("#a42b3f", "#bf3d53"),
            command=self._clear
        ).grid(row=0, column=1)

        self.progress = ctk.CTkProgressBar(left, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.progress.grid_remove()

        # Tip
        tip = ctk.CTkFrame(left, fg_color=("#162022", "#1c292f"),
                            corner_radius=18, border_width=1, border_color=("#224b65", "#1a2d3d"))
        tip.grid(row=3, column=0, sticky="ew")
        ctk.CTkLabel(tip,
            text="Tip: Pick soil & climate for quick presets, or enter values manually. Use Weather to auto-fill temp/humidity/rain.",
            font=ctk.CTkFont(family="Segoe UI", size=11), text_color="gray65",
            wraplength=400, justify="left"
        ).grid(row=0, column=0, sticky="w", padx=18, pady=14)

    def _build_right_panel(self):
        right = ctk.CTkFrame(self.main, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        # Result card
        res = ctk.CTkFrame(right, fg_color=("#171f23", "#1f2630"),
                            corner_radius=20, border_width=1, border_color=("#2b5065", "#1f3242"))
        res.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        res.grid_columnconfigure(0, weight=1)

        self.res_lbl = ctk.CTkLabel(res, text="🌿 Recommended Crop: —",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#d7ffd9", wraplength=400, justify="left")
        self.res_lbl.grid(row=0, column=0, sticky="w", padx=18, pady=(16, 4))

        self.conf_lbl = ctk.CTkLabel(res, text="",
            font=ctk.CTkFont(family="Segoe UI", size=12), text_color="#9bb8c8")
        self.conf_lbl.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 12))

        # Top predictions
        self.top_frame = ctk.CTkFrame(right, fg_color="transparent")
        self.top_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_remove()

        ctk.CTkLabel(self.top_frame, text="📊 Top Predictions",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"), anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(0, 6))

        self.top_rows = []
        for i in range(5):
            rf = ctk.CTkFrame(self.top_frame, fg_color="transparent", height=20)
            rf.grid(row=i+1, column=0, sticky="ew", pady=1)
            rf.grid_columnconfigure(1, weight=1)
            n = ctk.CTkLabel(rf, text="", font=ctk.CTkFont(size=11, weight="bold"), width=120, anchor="w")
            n.grid(row=0, column=0, sticky="w")
            b = ctk.CTkProgressBar(rf, height=12, corner_radius=6,
                fg_color=("#2a3a3f", "#25333a"), progress_color=("#3daf6b", "#45c47a"))
            b.grid(row=0, column=1, sticky="ew", padx=(4, 4))
            b.set(0)
            p = ctk.CTkLabel(rf, text="", font=ctk.CTkFont(size=10), width=50, anchor="e")
            p.grid(row=0, column=2, sticky="e")
            self.top_rows.append((n, b, p, rf))

        # Explanation
        exp = ctk.CTkFrame(right, fg_color=("#171f23", "#1f2630"),
                            corner_radius=20, border_width=1, border_color=("#2b5065", "#1f3242"))
        exp.grid(row=2, column=0, sticky="nsew")
        exp.grid_columnconfigure(0, weight=1)
        exp.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(exp, text="💡 Why this crop?",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(14, 6))

        self.exp_text = ctk.CTkTextbox(exp, height=140, corner_radius=12, border_width=0,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color=("#1d262f", "#222d38"), text_color=("#d4dde8", "#e4edf5"),
            wrap="word", activate_scrollbars=True)
        self.exp_text.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.exp_text.insert("0.0", "Enter soil parameters and click Predict to see the explanation.")
        self.exp_text.configure(state="disabled")

        # Chat
        chat = ctk.CTkFrame(right, fg_color=("#171f23", "#1f2630"),
                             corner_radius=20, border_width=1, border_color=("#2b5065", "#1f3242"))
        chat.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        chat.grid_columnconfigure(0, weight=1)
        chat.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(chat, text="💬 Follow-up Chat",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(14, 6))

        self.chat_text = ctk.CTkTextbox(chat, height=160, corner_radius=12, border_width=0,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color=("#1d262f", "#222d38"), text_color=("#d4dde8", "#e4edf5"),
            wrap="word", activate_scrollbars=True)
        self.chat_text.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 8))
        self.chat_text.insert("0.0", "Ask a follow-up question after a prediction.")
        self.chat_text.configure(state="disabled")

        ci = ctk.CTkFrame(chat, fg_color="transparent")
        ci.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        ci.grid_columnconfigure(0, weight=1)

        self.chat_entry = ctk.CTkEntry(ci, placeholder_text="Ask a follow-up question...",
            height=38, corner_radius=10, border_width=2, font=ctk.CTkFont(family="Segoe UI", size=11))
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.chat_entry.bind('<Return>', lambda e: self._chat())

        ctk.CTkButton(ci, text="Send", width=90, height=38, corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), command=self._chat
        ).grid(row=0, column=1)

    # ============ HANDLERS ============
    def _on_soil_change(self, choice):
        if choice == "-- Manual --" or choice not in [v['name'] for v in SOIL_PROFILES.values()]:
            return
        # Find profile
        profile = next(v for v in SOIL_PROFILES.values() if v['name'] == choice)
        self.entries['N'].delete(0, "end"); self.entries['N'].insert(0, str(profile['N']))
        self.entries['P'].delete(0, "end"); self.entries['P'].insert(0, str(profile['P']))
        self.entries['K'].delete(0, "end"); self.entries['K'].insert(0, str(profile['K']))
        self.entries['ph'].delete(0, "end"); self.entries['ph'].insert(0, str(profile['ph']))

    def _on_climate_change(self, choice):
        if choice == "-- Manual --" or choice not in [v['name'] for v in CLIMATE_PRESETS.values()]:
            return
        preset = next(v for v in CLIMATE_PRESETS.values() if v['name'] == choice)
        self.entries['temperature'].delete(0, "end"); self.entries['temperature'].insert(0, str(preset['temperature']))
        self.entries['humidity'].delete(0, "end"); self.entries['humidity'].insert(0, str(preset['humidity']))
        self.entries['rainfall'].delete(0, "end"); self.entries['rainfall'].insert(0, str(preset['rainfall']))

    def _fetch_weather(self):
        city = self.city_entry.get().strip()
        if not city:
            messagebox.showwarning("Weather", "Enter a city name first.")
            return
        self.predict_btn.configure(state="disabled", text="⏳ Fetching...")
        def worker():
            wx = fetch_weather(city)
            def done():
                self.predict_btn.configure(state="normal", text="🌾 Predict Crop")
                if wx is None:
                    messagebox.showerror("Weather", "Could not fetch weather. Check city name or connection.")
                    return
                self.entries['temperature'].delete(0, "end"); self.entries['temperature'].insert(0, f"{wx['temperature']:.1f}")
                self.entries['humidity'].delete(0, "end"); self.entries['humidity'].insert(0, f"{wx['humidity']:.0f}")
                self.entries['rainfall'].delete(0, "end"); self.entries['rainfall'].insert(0, f"{wx['rainfall']:.1f}")
                self._chat_append("System", f"☁ Weather from {wx['city']}: {wx['temperature']}°C, {wx['humidity']}% humidity, {wx['rainfall']}mm rain.")
            self.root.after(0, done)
        threading.Thread(target=worker, daemon=True).start()

    def _validate(self):
        vals = []
        for key, (lo, hi) in INPUT_RANGES.items():
            v = self.entries[key].get().strip()
            if not v:
                messagebox.showwarning("Missing", f"Enter {key}.")
                return None
            try:
                n = float(v)
            except ValueError:
                messagebox.showerror("Invalid", f"{key} must be a number.")
                return None
            if n < lo or n > hi:
                messagebox.showwarning("Range", f"{key} should be {lo}–{hi}.")
            vals.append(n)
        return vals

    def _predict(self):
        vals = self._validate()
        if vals is None:
            return

        X = np.array(vals).reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        scores = [(self.le.inverse_transform([i])[0], probs[i]) for i in range(len(probs))]
        scores.sort(key=lambda x: -x[1])

        crop, conf = scores[0]
        self.res_lbl.configure(text=f"🌿 Recommended Crop: {crop.upper()}")
        self.conf_lbl.configure(text=f"Confidence: {conf*100:.1f}%")
        if conf > 0.8: self.res_lbl.configure(text_color=("#4ade80", "#4ade80"))
        elif conf > 0.6: self.res_lbl.configure(text_color=("#fb923c", "#fb923c"))
        else: self.res_lbl.configure(text_color=("#f87171", "#f87171"))

        self.prediction_ctx = {'crop': crop, 'vals': vals, 'conf': conf*100, 'top5': scores[:5]}
        self._render_top(scores[:5])
        self._clear_chat()

        self.progress.grid()
        self.progress.start()
        self.predict_btn.configure(state="disabled")
        threading.Thread(target=self._explain, args=(crop, vals, conf*100), daemon=True).start()

    def _render_top(self, scores):
        for i, (n, b, p, f) in enumerate(self.top_rows):
            if i < len(scores):
                c, s = scores[i]
                n.configure(text=f"🌱 {c.title()}")
                b.set(float(s))
                p.configure(text=f"{s*100:.0f}%")
                f.grid()
            else:
                f.grid_remove()
        self.top_frame.grid() if scores else self.top_frame.grid_remove()

    def _explain(self, crop, vals, conf):
        if self.llm:
            try:
                cr = self.crop_ranges.get(crop.lower(), {})
                parts = []
                for k in FEATURES:
                    r = cr.get(k)
                    if r and isinstance(r, (list, tuple)) and len(r) == 2:
                        parts.append(f"{k}:{r[0]}-{r[1]}")
                prompt = (
                    f"You are a concise agricultural assistant. Explain why {crop} is recommended.\n"
                    f"Crop ranges: {', '.join(parts) if parts else 'unavailable'}\n"
                    f"User values: N={vals[0]}, P={vals[1]}, K={vals[2]}, "
                    f"Temp={vals[3]}°C, Hum={vals[4]}%, pH={vals[5]}, Rain={vals[6]}mm\n"
                    f"Confidence: {conf:.0f}%\n"
                    f"Reply in ≤3 sentences. Start with 'Did you know'."
                )
                resp = self.llm(prompt, max_tokens=120, temperature=0.35, stop=['\n'])
                txt = ""
                if isinstance(resp, dict):
                    txt = resp.get('choices', [{}])[0].get('text', '') or ''
                else:
                    txt = getattr(resp, 'choices', [{}])[0].get('text', '') or ''
                txt = txt.strip()
                if txt:
                    txt = re.sub(r'\s{2,}', ' ', txt)
                    if not re.search(r'[.!?]$', txt):
                        txt += '.'
                    if 'did you know' not in txt.lower():
                        txt = f"Did you know {txt}"
            except Exception:
                txt = rule_explanation(crop, vals)
        else:
            txt = rule_explanation(crop, vals)

        fert = self._fertilizer_advice(crop, vals)
        self.root.after(0, lambda: self._finish_explain(txt, fert))

    def _finish_explain(self, txt, fert):
        self.progress.stop()
        self.progress.grid_remove()
        self.predict_btn.configure(state="normal")

        self.exp_text.configure(state="normal")
        self.exp_text.delete("0.0", "end")
        self.exp_text.insert("0.0", txt)
        self.exp_text.configure(state="disabled")

        self._chat_append("Assistant", txt)
        if fert:
            self._chat_append("System", fert)
        self.chat_history = [("assistant", txt)]

    def _fertilizer_advice(self, crop, vals):
        cr = self.crop_ranges.get(crop.lower(), {})
        if not cr:
            return ""
        lines = ["═══ Fertilizer Advice ═══"]
        for k, v, name in zip(['N', 'P', 'K'], vals[:3], ['Nitrogen', 'Phosphorus', 'Potassium']):
            r = cr.get(k)
            if not r or len(r) != 2:
                continue
            lo, hi = r
            if v < lo:
                lines.append(f"{name}: add {lo - v:.0f} kg/ha (have {v:.0f}, need {lo:.0f}-{hi:.0f})")
            elif v > hi:
                lines.append(f"{name}: reduce {v - hi:.0f} kg/ha (have {v:.0f}, excess)")
            else:
                lines.append(f"{name}: OK ({v:.0f} in range {lo:.0f}-{hi:.0f})")
        return "\n".join(lines)

    def _chat(self):
        q = self.chat_entry.get().strip()
        if not q:
            return
        if not self.prediction_ctx:
            messagebox.showwarning("No Prediction", "Predict a crop first.")
            return
        self.chat_entry.delete(0, "end")
        self.chat_history.append(("user", q))
        self._chat_append("You", q)
        threading.Thread(target=self._chat_response, args=(q,), daemon=True).start()

    def _chat_response(self, q):
        if not self.llm:
            self.root.after(0, lambda: self._chat_append("Assistant", "LLM unavailable. Install llama-cpp-python and add a .gguf model to LLM/."))
            return
        ctx = self.prediction_ctx
        v = dict(zip(FEATURES, ctx['vals']))
        hist = "\n".join([f"{r.title()}: {t}" for r, t in self.chat_history[-6:]])
        prompt = (
            f"You are a concise agricultural assistant. The user got {ctx['crop']} recommendation.\n"
            f"Inputs: N={v['N']}, P={v['P']}, K={v['K']}, Temp={v['temperature']}°C, "
            f"Hum={v['humidity']}%, pH={v['ph']}, Rain={v['rainfall']}mm\n"
            f"Crop ranges:\n{self.crop_summary}\n\n"
            f"Chat:\n{hist}\nUser: {q}\nAssistant:"
        )
        try:
            resp = self.llm(prompt, max_tokens=150, temperature=0.35, stop=['\n\n', '\nUser:', '\nAssistant:'])
            ans = ""
            if isinstance(resp, dict):
                ans = resp.get('choices', [{}])[0].get('text', '') or ''
            else:
                ans = getattr(resp, 'choices', [{}])[0].get('text', '') or ''
            ans = ans.strip()
            ans = re.sub(r'\s{2,}', ' ', ans)
            if not ans:
                ans = "I couldn't produce a clear answer. Try rephrasing."
            if not re.search(r'[.!?]$', ans):
                ans += '.'
        except Exception as e:
            ans = f"Chat error: {e}"
        self.chat_history.append(("assistant", ans))
        self.root.after(0, lambda: self._chat_append("Assistant", ans))

    def _chat_append(self, speaker, msg):
        self.chat_text.configure(state="normal")
        content = self.chat_text.get("0.0", "end").strip()
        if content in ("Ask a follow-up question after a prediction.", "Enter soil parameters and click Predict to see the explanation."):
            self.chat_text.delete("0.0", "end")
        self.chat_text.insert("end", f"{speaker}: {msg}\n\n")
        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def _clear_chat(self):
        self.chat_history = []
        self.chat_text.configure(state="normal")
        self.chat_text.delete("0.0", "end")
        self.chat_text.insert("0.0", "Ask a follow-up question after a prediction.")
        self.chat_text.configure(state="disabled")

    def _clear(self):
        for e in self.entries.values():
            e.delete(0, "end")
        self.soil_var.set("-- Manual --")
        self.climate_var.set("-- Manual --")
        self.city_entry.delete(0, "end")
        self.res_lbl.configure(text="🌿 Recommended Crop: —", text_color=("gray40", "gray70"))
        self.conf_lbl.configure(text="")
        self.exp_text.configure(state="normal")
        self.exp_text.delete("0.0", "end")
        self.exp_text.insert("0.0", "Enter soil parameters and click Predict to see the explanation.")
        self.exp_text.configure(state="disabled")
        self._clear_chat()
        self._clear_top()
        self.prediction_ctx = None

    def _clear_top(self):
        for n, b, p, f in self.top_rows:
            n.configure(text=""); b.set(0); p.configure(text=""); f.grid_remove()
        self.top_frame.grid_remove()


# ============ MAIN ============
def main():
    print("🌱 Initializing SeedBrain...")
    model, le, acc = load_model()
    if model is None:
        print("Failed to load model.")
        return
    print(f"✅ Model loaded (acc: {acc*100:.1f}%) — {len(le.classes_)} crops")

    print("🤖 Loading LLM...")
    llm = load_llm()
    if llm: print("✅ LLM loaded")
    else: print("⚠️ LLM unavailable (optional)")

    print("🚀 Launching GUI...")
    root = ctk.CTk()
    app = CropApp(root, model, le, acc, llm=llm)
    root.mainloop()


if __name__ == "__main__":
    main()