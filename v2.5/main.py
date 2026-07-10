import json
import os
import sys
import threading
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox, simpledialog
import warnings
warnings.filterwarnings('ignore')

try:
    import customtkinter as ctk
except ImportError:
    raise ImportError("customtkinter is required. Install it with: pip install customtkinter")

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from scripts.weather import fetch_weather

# ============ CONFIGURATION ============
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

CROP_RANGES_PATH = BASE_DIR / 'crop_ranges.json'
SOIL_PROFILES_PATH = BASE_DIR / 'soil_profiles.json'
CROP_REGION_MAP_PATH = BASE_DIR / 'crop_region_map.json'
LLM_MODEL_DIR = BASE_DIR / 'LLM'
LLM_CONTEXT_WINDOW = 8000

# ============ TRANSLATIONS ============
LANGUAGES = ["en", "fr"]

STRINGS = {
    "app_title": {"en": "Crop Recommendation System", "fr": "Système de Recommandation de Cultures"},
    "app_subtitle": {"en": "AI-powered soil and climate analysis for smarter crop selection.", "fr": "Analyse IA du sol et du climat pour des choix de cultures plus intelligents."},
    "invoke_info": {"en": "Enter values on the left, then get a crop recommendation. Ask follow-up questions in the conversation panel.", "fr": "Saisissez les valeurs à gauche, puis obtenez une recommandation. Posez des questions dans le panneau de conversation."},
    "model_accuracy": {"en": "Model Accuracy", "fr": "Précision du modèle"},
    "input_title": {"en": "Soil & Climate Inputs", "fr": "Paramètres du sol et du climat"},
    "soil_type": {"en": "Soil Type (auto-fills NPK/pH)", "fr": "Type de sol (remplit NPK/pH)"},
    "region": {"en": "Region (filter + weather)", "fr": "Région (filtre + météo)"},
    "manual_entry": {"en": "-- Manual entry --", "fr": "-- Saisie manuelle --"},
    "all_regions": {"en": "All Regions", "fr": "Toutes les régions"},
    "continent": {"en": "Continent", "fr": "Continent"},
    "predict_btn": {"en": "Predict Crop", "fr": "Prédire"},
    "clear_btn": {"en": "Clear All", "fr": "Tout effacer"},
    "autofill_btn": {"en": "Describe field", "fr": "Décrire le champ"},
    "conversation_header": {"en": "Conversation", "fr": "Conversation"},
    "recommended_crop": {"en": "Recommended Crop", "fr": "Culture recommandée"},
    "confidence": {"en": "Confidence", "fr": "Confiance"},
    "fetch_weather_btn": {"en": "Fetch Weather", "fr": "Météo"},
    "send_btn": {"en": "Send", "fr": "Envoyer"},
    "chat_placeholder": {"en": "Ask a follow-up question...", "fr": "Posez une question..."},
    "chat_placeholder_empty": {"en": "Enter soil parameters, click Predict, then ask follow-up questions here.", "fr": "Entrez les paramètres du sol, cliquez sur Prédire, puis posez vos questions."},
    "llm_loading": {"en": "LLM status: checking...", "fr": "Statut LLM : vérification..."},
    "llm_loaded": {"en": "LLM loaded", "fr": "LLM chargé"},
    "llm_unavailable": {"en": "LLM unavailable", "fr": "LLM indisponible"},
    "llm_chat_unavailable": {"en": "LLM chat unavailable. Install llama-cpp-python and place a GGUF model in the LLM folder.", "fr": "Chat LLM indisponible. Installez llama-cpp-python et placez un modèle GGUF dans le dossier LLM."},
    "llm_explain_unavailable": {"en": "LLM explanation unavailable. Install llama-cpp-python and place a GGUF model in the LLM folder.", "fr": "Explication LLM indisponible. Installez llama-cpp-python et placez un modèle GGUF dans le dossier LLM."},
    "no_prediction": {"en": "No Prediction", "fr": "Aucune prédiction"},
    "no_prediction_msg": {"en": "Please generate a crop prediction before asking follow-up questions.", "fr": "Veuillez générer une prédiction avant de poser des questions."},
    "select_region": {"en": "Select Region", "fr": "Sélectionnez une région"},
    "select_region_msg": {"en": "Please select a specific region before fetching weather.", "fr": "Veuillez sélectionner une région avant de récupérer la météo."},
    "weather_error": {"en": "Weather Error", "fr": "Erreur météo"},
    "weather_error_msg": {"en": "Could not fetch weather data. Check your internet connection.", "fr": "Impossible de récupérer les données météo. Vérifiez votre connexion."},
    "missing_input": {"en": "Missing Input", "fr": "Saisie manquante"},
    "missing_input_msg": {"en": "Please enter a value for", "fr": "Veuillez entrer une valeur pour"},
    "invalid_input": {"en": "Invalid Input", "fr": "Saisie invalide"},
    "invalid_input_msg": {"en": "must be a number between", "fr": "doit être un nombre entre"},
    "adjusted_input": {"en": "Adjusted Input", "fr": "Valeur ajustée"},
    "adjusted_input_msg": {"en": "was adjusted to the safe range", "fr": "a été ajusté à la plage sécurisée"},
    "prediction_error": {"en": "Prediction Error", "fr": "Erreur de prédiction"},
    "prediction_error_msg": {"en": "An error occurred during prediction", "fr": "Une erreur est survenue lors de la prédiction"},
    "tip_text": {"en": "Tip: Use accurate soil data. Select a region to filter suitable crops and fetch local weather.", "fr": "Conseil : Utilisez des données précises. Sélectionnez une région pour filtrer les cultures et obtenir la météo."},
    "assistant_label": {"en": "Assistant", "fr": "Assistant"},
    "you_label": {"en": "You", "fr": "Vous"},
    "system_label": {"en": "System", "fr": "Système"},
    "fertilizer_header": {"en": "Fertilizer Advice", "fr": "Conseil en fertilisation"},
    "top_predictions_header": {"en": "Top Predictions", "fr": "Meilleures prédictions"},
    "autofill_title": {"en": "Describe Your Field", "fr": "Décrivez votre champ"},
    "autofill_prompt": {"en": "Describe your soil, climate, and growing conditions in natural language. The AI will extract values and fill the input fields.", "fr": "Décrivez votre sol, votre climat et vos conditions de culture en langage naturel. L'IA extraira les valeurs et remplira les champs."},
    "autofill_placeholder": {"en": "e.g., I have clay soil in Nigeria, hot and humid with lots of rain...", "fr": "Ex: sol argileux au Maroc, chaud et humide avec beaucoup de pluie..."},
    "autofill_fill_btn": {"en": "Auto-fill Fields", "fr": "Remplir les champs"},
    "autofill_cancel_btn": {"en": "Cancel", "fr": "Annuler"},
    "autofill_error": {"en": "Could not parse field description. Try being more specific.", "fr": "Impossible d'analyser la description. Soyez plus précis."},
    "autofill_no_llm": {"en": "LLM is not loaded. Cannot auto-fill fields.", "fr": "LLM non chargé. Impossible de remplir les champs."},
    "autofill_success": {"en": "Field auto-filled from description", "fr": "Champs remplis depuis la description"},
    "fert_add": {"en": "add", "fr": "ajouter"},
    "fert_reduce": {"en": "reduce by", "fr": "réduire de"},
    "fert_ok": {"en": "OK", "fr": "OK"},
    "fert_you_have": {"en": "you have", "fr": "vous avez"},
    "fert_need": {"en": "need", "fr": "besoin"},
    "fert_excess": {"en": "excess", "fr": "excès"},
    "weather_autofilled": {"en": "Weather auto-filled from", "fr": "Météo récupérée depuis"},
}

LLM_SYSTEM_PROMPTS = {
    "en": (
        "You are a concise crop recommendation assistant. "
        "Keep responses brief, factual, and no more than 3 sentences. "
        "Do not use markdown or HTML formatting."
    ),
    "fr": (
        "Vous êtes un assistant concis de recommandation de cultures. "
        "Répondez brièvement, factuellement, en 3 phrases maximum. "
        "N'utilisez pas le formatage markdown ou HTML."
    ),
}

LLM_AUTOFILL_PROMPTS = {
    "en": (
        "Extract soil and climate values from this field description. "
        "Respond ONLY with a valid JSON object (no markdown, no explanation):\n"
        '{"N": number, "P": number, "K": number, "organic_carbon": number, '
        '"temperature": number, "humidity": number, "ph": number, "rainfall": number}\n\n'
        "Constraints:\n"
        "- N: 0-70 mg/kg, P: 0-25 mg/kg, K: 0-3000 mg/kg\n"
        "- organic_carbon: 0-650 g/kg\n"
        "- temperature: 0-45 C, humidity: 5-100 percent\n"
        "- ph: 2.0-11.0, rainfall: 0-5 mm (daily forecast)\n"
        "Infer missing values from soil type, climate zone, or region. "
        "All values must be within the ranges above."
    ),
    "fr": (
        "Extrayez les valeurs du sol et du climat à partir de cette description de champ. "
        "Répondez UNIQUEMENT avec un objet JSON valide (pas de markdown, pas d'explication) :\n"
        '{"N": number, "P": number, "K": number, "organic_carbon": number, '
        '"temperature": number, "humidity": number, "ph": number, "rainfall": number}\n\n'
        "Contraintes :\n"
        "- N : 0-70 mg/kg, P : 0-25 mg/kg, K : 0-3000 mg/kg\n"
        "- organic_carbon : 0-650 g/kg\n"
        "- température : 0-45 C, humidité : 5-100 pourcent\n"
        "- pH : 2.0-11.0, précipitations : 0-5 mm (prévisions quotidiennes)\n"
        "Déduisez les valeurs manquantes du type de sol, de la zone climatique ou de la région. "
        "Toutes les valeurs doivent être dans les plages ci-dessus."
    ),
}

NUTRIENT_NAMES = {
    "en": {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium"},
    "fr": {"N": "Azote", "P": "Phosphore", "K": "Potassium"},
}

NUTRIENT_NAMES_FULL = {
    'organic_carbon': ('Organic Carbon (g/kg)', 'Carbone organique (g/kg)'),
}

FEATURES = ['N', 'P', 'K', 'organic_carbon', 'temperature', 'humidity', 'ph', 'rainfall']

FIELD_LABELS = {
    "en": [
        ("Nitrogen (N)", "N", "0 \u2013 55 kg/ha"),
        ("Phosphorus (P)", "P", "5 \u2013 25 kg/ha"),
        ("Potassium (K)", "K", "5 \u2013 200 kg/ha"),
        ("Organic Carbon (g/kg)", "organic_carbon", "0 \u2013 650 g/kg"),
        ("Temperature", "temperature", "0 \u2013 40 \u00b0C"),
        ("Humidity", "humidity", "5 \u2013 100 %"),
        ("pH Level", "ph", "2.0 \u2013 11.0"),
        ("Rainfall", "rainfall", "0 \u2013 16 mm/day"),
    ],
    "fr": [
        ("Azote (N)", "N", "0 \u2013 55 kg/ha"),
        ("Phosphore (P)", "P", "5 \u2013 25 kg/ha"),
        ("Potassium (K)", "K", "5 \u2013 200 kg/ha"),
        ("Carbone organique (g/kg)", "organic_carbon", "0 \u2013 650 g/kg"),
        ("Temp\u00e9rature", "temperature", "0 \u2013 40 \u00b0C"),
        ("Humidit\u00e9", "humidity", "5 \u2013 100 %"),
        ("pH", "ph", "2.0 \u2013 11.0"),
        ("Pr\u00e9cipitations", "rainfall", "0 \u2013 16 mm/jour"),
    ],
}


# ============ LOAD THE SAVED MODEL ============
def train_model():
    """Load the unified XGBoost model (GAEZ + GROW-Africa) and label encoder."""
    model_path = BASE_DIR / 'xgboost_model.pkl'
    le_path = BASE_DIR / 'label_encoder.pkl'
    continent_enc_path = BASE_DIR / 'continent_encoder.pkl'

    if not model_path.exists():
        print("⏳ First-time setup: running data preparation and training...")
        setup_ok = run_setup_scripts()
        if not setup_ok:
            msg = ("Model files not found and auto-setup failed.\n"
                   "Run: python3 scripts/prepare_data.py && python3 scripts/train_model.py")
            messagebox.showerror("Error", msg)
            return None, None, None, None

    try:
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        continent_le = joblib.load(continent_enc_path) if continent_enc_path.exists() else None
    except Exception as exc:
        messagebox.showerror("Error", f"Failed to load model:\n{exc}")
        return None, None, None, None

    dataset_path = BASE_DIR / 'dataset' / 'gov_dataset.csv'
    if not dataset_path.exists():
        dataset_path = BASE_DIR / 'dataset' / 'Crop_recommendation_merged.csv'
    if not dataset_path.exists():
        dataset_path = BASE_DIR / 'dataset' / 'Crop_recommendation.csv'

    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        return model, le, continent_le, 0.0

    df = df.drop_duplicates()
    features = ['N', 'P', 'K', 'organic_carbon', 'temperature', 'humidity', 'ph', 'rainfall']
    if continent_le is not None:
        df['continent_encoded'] = continent_le.transform(df['continent'])
        X = df[features + ['continent_encoded']].values
    else:
        X = df[features].values

    try:
        y_encoded = le.transform(df['label'])
        np.random.seed(42)
        _, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        ranges = X.max(axis=0) - X.min(axis=0)
        noise = np.random.normal(0, 1, X_test.shape) * ranges * 0.01
        accuracy = model.score(X_test + noise, y_test)
    except Exception:
        accuracy = 0.0
    return model, le, continent_le, accuracy


def run_setup_scripts():
    """Run data preparation and training scripts on first launch."""
    try:
        import subprocess
        prep = subprocess.run(
            [sys.executable, str(BASE_DIR / 'scripts' / 'prepare_data.py')],
            capture_output=True, text=True, cwd=BASE_DIR
        )
        if prep.returncode != 0:
            print("prepare_data.py failed:", prep.stderr)
            return False
        train = subprocess.run(
            [sys.executable, str(BASE_DIR / 'scripts' / 'train_model.py')],
            capture_output=True, text=True, cwd=BASE_DIR
        )
        if train.returncode != 0:
            print("train_model.py failed:", train.stderr)
            return False
        return True
    except Exception as exc:
        print(f"Auto-setup failed: {exc}")
        return False


def find_gguf_model():
    models = sorted(LLM_MODEL_DIR.glob('*.gguf'))
    if not models:
        return None
    return models[0]

def load_llm_model():
    model_path = find_gguf_model()
    if model_path is None:
        print("⚠️ No .gguf model found in LLM/ folder. LLM explanations disabled.")
        return None
    print(f"LLM model path: {model_path}")
    if Llama is None:
        print("⚠️ llama_cpp is not installed. LLM explanations are disabled.")
        return None

    if not model_path.exists():
        print(f"⚠️ LLM model file not found at: {model_path}")
        return None

    try:
        llm = Llama(
            model_path=str(model_path),
            n_threads=min(4, os.cpu_count() or 1),
            n_ctx=LLM_CONTEXT_WINDOW
        )
        print("✅ LLM is loaded successfully.")
        return llm
    except Exception as exc:
        print(f"⚠️ Failed to load LLM model: {exc}")
        return None


def load_crop_ranges_json():
    if not CROP_RANGES_PATH.exists():
        return None
    try:
        with open(CROP_RANGES_PATH, 'r', encoding='utf-8') as f:
            raw_ranges = json.load(f)
            return {key.lower(): value for key, value in raw_ranges.items()}
    except Exception:
        return None


def build_crop_range_summary():
    crop_ranges = load_crop_ranges_json()
    if crop_ranges is not None:
        lines = []
        for crop in sorted(crop_ranges.keys()):
            values = crop_ranges[crop]
            line = (
                f"{crop}: N {values['N'][0]:.0f}-{values['N'][1]:.0f}, "
                f"P {values['P'][0]:.0f}-{values['P'][1]:.0f}, "
                f"OC {values['organic_carbon'][0]:.0f}-{values['organic_carbon'][1]:.0f}, "
                f"K {values['K'][0]:.0f}-{values['K'][1]:.0f}, "
                f"Temp {values['temperature'][0]:.1f}-{values['temperature'][1]:.1f}, "
                f"Humidity {values['humidity'][0]:.1f}-{values['humidity'][1]:.1f}, "
                f"pH {values['ph'][0]:.2f}-{values['ph'][1]:.2f}, "
                f"Rain {values['rainfall'][0]:.1f}-{values['rainfall'][1]:.1f}"
            )
            lines.append(line)
        return "\n".join(lines)

    try:
        df = pd.read_csv(BASE_DIR / 'dataset' / 'Crop_recommendation_merged.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv(BASE_DIR / 'dataset' / 'Crop_recommendation.csv')
        except FileNotFoundError:
            return "Crop range data unavailable."

    lines = []
    for crop in sorted(df['label'].unique()):
        crop_df = df[df['label'] == crop]
        line = (
            f"{crop}: N {crop_df['N'].min():.0f}-{crop_df['N'].max():.0f}, "
            f"P {crop_df['P'].min():.0f}-{crop_df['P'].max():.0f}, "
            f"OC {crop_df['organic_carbon'].min():.0f}-{crop_df['organic_carbon'].max():.0f}, "
            f"K {crop_df['K'].min():.0f}-{crop_df['K'].max():.0f}, "
            f"Temp {crop_df['temperature'].min():.1f}-{crop_df['temperature'].max():.1f}, "
            f"Humidity {crop_df['humidity'].min():.1f}-{crop_df['humidity'].max():.1f}, "
            f"pH {crop_df['ph'].min():.2f}-{crop_df['ph'].max():.2f}, "
            f"Rain {crop_df['rainfall'].min():.1f}-{crop_df['rainfall'].max():.1f}"
        )
        lines.append(line)

    return "\n".join(lines)


def load_soil_profiles():
    if not SOIL_PROFILES_PATH.exists():
        return None
    try:
        with open(SOIL_PROFILES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_crop_region_map():
    if not CROP_REGION_MAP_PATH.exists():
        return None
    try:
        with open(CROP_REGION_MAP_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def rule_based_explanation(crop, input_values, confidence):
    cr = load_crop_ranges_json() or {}
    single = cr.get(crop.lower(), {})
    keys = ['N', 'P', 'K', 'organic_carbon', 'temperature', 'humidity', 'ph', 'rainfall']
    labels = {
        'N': 'nitrogen', 'P': 'phosphorus', 'K': 'potassium',
        'organic_carbon': 'organic carbon',
        'temperature': 'temperature', 'humidity': 'humidity',
        'ph': 'pH', 'rainfall': 'rainfall'
    }

    environment_facts = {
        'beans': 'it benefits from warm soil and moderate rainfall',
        'cassava': 'it tolerates poor soils and dry conditions once established',
        'sorghum': 'it thrives in warm, well-drained soils with moderate rainfall',
        'maize': 'it prefers warm, sunny weather and well-drained soil with moderate rainfall',
        'orange': 'it prefers warm, subtropical weather and well-drained soil',
        
        'rice': 'it thrives in warm, moist fields with steady water availability',
        'soybean': 'it does well in warm, well-drained soils with consistent moisture',
        'tomatoes': 'it prefers warm, sunny conditions with well-drained, fertile soil',
        'yam': 'it thrives in warm, humid conditions with loose, well-drained soil',
    }

    matched = []
    for k, val in zip(keys, input_values):
        rng = single.get(k)
        if not rng or len(rng) != 2:
            continue
        try:
            low, high = float(rng[0]), float(rng[1])
        except Exception:
            continue
        if low > high or (low == 0 and high == 0):
            continue
        try:
            v = float(val)
        except Exception:
            continue
        if low <= v <= high:
            matched.append((k, low, high))

    fact = environment_facts.get(crop.lower(),
        'it is suited to these conditions')

    if confidence >= 80:
        if len(matched) == len(keys):
            return (f"{crop.title()} is a strong match. All your input values fall within "
                    f"its ideal ranges, and {fact}.")
        elif matched:
            names = [labels[k] for k, _, _ in matched[:3]]
            vals = ', '.join(names)
            return (f"{crop.title()} is a good fit. Your {vals} are within ideal range, "
                    f"and {fact}.")
        else:
            return (f"{crop.title()} is recommended with {confidence:.0f}% confidence. "
                    f"{fact}.")
    elif confidence >= 50:
        return (f"{crop.title()} may be suitable (confidence: {confidence:.0f}%). "
                f"Some of your values are near the edge of its ideal range. {fact}.")
    else:
        return (f"Results are uncertain (confidence: {confidence:.0f}%). "
                f"Your conditions don't strongly match any crop. Consider soil testing "
                f"for more accurate NPK values.")


# ============ GUI APPLICATION ============
class CropRecommendationApp:
    def __init__(self, root, model, label_encoder, continent_encoder, accuracy, llm=None):
        self.root = root
        self.model = model
        self.label_encoder = label_encoder
        self.continent_encoder = continent_encoder
        self.accuracy = accuracy
        self.llm = llm
        self.crop_ranges_summary = build_crop_range_summary()
        self.soil_profiles = load_soil_profiles()
        self.crop_region_map = load_crop_region_map()
        self.lang = "en"

        self.root.title(f"\U0001f33e {self._t('app_title')}")
        self.root.geometry("900x850")
        self.root.minsize(800, 750)

        # Configure grid weights for responsiveness
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Main scrollable container
        self.main_scroll = ctk.CTkScrollableFrame(self.root, fg_color=("#121212", "#1a1a1a"), border_width=0)
        self.main_scroll.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.main_scroll.grid_columnconfigure((0, 1), weight=1, uniform="col")
        self.main_scroll.grid_rowconfigure(2, weight=1)

        # ========== HERO HEADER ==========
        self.hero_card = ctk.CTkFrame(
            self.main_scroll,
            fg_color=("#19232d", "#1e2d39"),
            corner_radius=22,
            border_width=1,
            border_color=("#34596f", "#1c3446")
        )
        self.hero_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18), padx=(0, 0))
        self.hero_card.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.hero_card,
            text=f"\U0001f331 {self._t('app_title')}",
            font=ctk.CTkFont(family="Segoe UI", size=30, weight="bold"),
            text_color=("#d7ffd9", "#d7ffd9")
        )
        self.title_label.grid(row=0, column=0, sticky="w", padx=24, pady=(24, 6))

        self.subtitle_label = ctk.CTkLabel(
            self.hero_card,
            text=self._t('app_subtitle'),
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="gray70",
            wraplength=860,
            justify="left"
        )
        self.subtitle_label.grid(row=1, column=0, sticky="w", padx=24, pady=(0, 20))

        self.status_row = ctk.CTkFrame(self.hero_card, fg_color="transparent")
        self.status_row.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 18))
        self.status_row.grid_columnconfigure(0, weight=1)
        self.status_row.grid_columnconfigure(1, weight=0)

        self.accuracy_badge = ctk.CTkFrame(
            self.status_row,
            fg_color=("#203d2f", "#1e3730"),
            corner_radius=18,
            border_width=1,
            border_color=("#2e6d4b", "#1b3f31")
        )
        self.accuracy_badge.grid(row=0, column=0, sticky="w")
        self.accuracy_label = ctk.CTkLabel(
            self.accuracy_badge,
            text=f"\U0001f3af {self._t('model_accuracy')}: {accuracy*100:.2f}%",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#98ffb5"
        )
        self.accuracy_label.pack(padx=18, pady=14)

        # Language selector
        self.lang_var = ctk.StringVar(value=self.lang)
        self.lang_menu = ctk.CTkOptionMenu(
            self.status_row,
            variable=self.lang_var,
            values=["EN", "FR"],
            command=self._on_lang_changed,
            height=28,
            width=56,
            corner_radius=14,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#1f3a4f", "#1a2e42"),
            button_color=("#2a546f", "#1d384d"),
            button_hover_color=("#3a7a9f", "#2a5a7f"),
        )
        self.lang_menu.grid(row=0, column=1, sticky="e", padx=(0, 10))

        self.llm_status_badge = ctk.CTkFrame(
            self.status_row,
            fg_color=("#22344f", "#1f2d43"),
            corner_radius=18,
            border_width=1,
            border_color=("#3a5a9e", "#1b2a3f")
        )
        self.llm_status_badge.grid(row=0, column=2, sticky="e")
        self.llm_status_label = ctk.CTkLabel(
            self.llm_status_badge,
            text=self._t('llm_loading'),
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#b9d2ff"
        )
        self.llm_status_label.pack(padx=18, pady=14)

        self.invoke_info = ctk.CTkLabel(
            self.hero_card,
            text=self._t('invoke_info'),
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="gray65",
            wraplength=860,
            justify="left"
        )
        self.invoke_info.grid(row=3, column=0, sticky="w", padx=24, pady=(0, 24))

        # ========== MAIN PANELS ==========
        self.left_panel = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        self.left_panel.grid_columnconfigure(0, weight=1)

        self.right_panel = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)

        # ========== INPUT CARD ==========
        self.input_card = ctk.CTkFrame(
            self.left_panel,
            fg_color=("#171f26", "#1e2931"),
            corner_radius=20,
            border_width=1,
            border_color=("#2f546a", "#1d3345")
        )
        self.input_card.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.input_card.grid_columnconfigure((0, 1), weight=1)

        self.input_title = ctk.CTkLabel(
            self.input_card,
            text=f"\U0001f9ea {self._t('input_title')}",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold")
        )
        self.input_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=22, pady=(22, 12))

        row_offset = 1
        self.soil_type_frame = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.soil_type_frame.grid(row=row_offset, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 6))
        self.soil_type_frame.grid_columnconfigure(0, weight=1)
        self.soil_type_frame.grid_columnconfigure(1, weight=1)
        self.soil_type_frame.grid_columnconfigure(2, weight=0)

        soil_lbl = ctk.CTkLabel(
            self.soil_type_frame,
            text=self._t('soil_type'),
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            anchor="w"
        )
        soil_lbl.grid(row=0, column=0, sticky="w", padx=(0, 10))

        region_lbl = ctk.CTkLabel(
            self.soil_type_frame,
            text=self._t('region'),
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            anchor="w"
        )
        region_lbl.grid(row=0, column=1, columnspan=2, sticky="w")

        soil_options = [self._t('manual_entry')]
        if self.soil_profiles:
            soil_options.extend(sorted(self.soil_profiles.keys()))

        self.soil_type_var = ctk.StringVar(value=soil_options[0])
        self.soil_type_menu = ctk.CTkOptionMenu(
            self.soil_type_frame,
            variable=self.soil_type_var,
            values=soil_options,
            command=self.on_soil_type_changed,
            height=36,
            corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11),
        )
        self.soil_type_menu.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(4, 10))

        region_options = [self._t('all_regions')]
        if self.crop_region_map:
            all_regions = set()
            for regions in self.crop_region_map.values():
                all_regions.update(regions)
            region_options.extend(sorted(all_regions))

        self.region_var = ctk.StringVar(value=region_options[0])
        self.region_menu = ctk.CTkOptionMenu(
            self.soil_type_frame,
            variable=self.region_var,
            values=region_options,
            height=36,
            corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11),
        )
        self.region_menu.grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=(4, 10))

        continent_lbl = ctk.CTkLabel(
            self.soil_type_frame,
            text=self._t('continent'),
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            anchor="w"
        )
        continent_lbl.grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(4, 0))

        if self.continent_encoder is not None:
            continent_options = list(self.continent_encoder.classes_)
        else:
            continent_options = ['global']
        self.continent_var = ctk.StringVar(value=continent_options[0])
        self.continent_menu = ctk.CTkOptionMenu(
            self.soil_type_frame,
            variable=self.continent_var,
            values=continent_options,
            height=36,
            corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=11),
        )
        self.continent_menu.grid(row=3, column=0, sticky="ew", padx=(0, 10), pady=(4, 10))

        self.weather_btn = ctk.CTkButton(
            self.soil_type_frame,
            text="\U0001f324",
            width=40,
            height=36,
            corner_radius=10,
            font=ctk.CTkFont(size=16),
            fg_color=("#1f4a6f", "#1e5075"),
            hover_color=("#2a6a9e", "#2a6a9e"),
            command=self.fetch_weather
        )
        self.weather_btn.grid(row=1, column=2, sticky="w", pady=(4, 10))

        self.entries = {}
        fields = FIELD_LABELS[self.lang]

        for i, (label, key, hint) in enumerate(fields):
            row = (i // 2) + row_offset + 1
            col = i % 2

            field_frame = ctk.CTkFrame(self.input_card, fg_color="transparent")
            field_frame.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
            field_frame.grid_columnconfigure(0, weight=1)

            lbl = ctk.CTkLabel(
                field_frame,
                text=label,
                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                anchor="w"
            )
            lbl.grid(row=0, column=0, sticky="w")

            entry = ctk.CTkEntry(
                field_frame,
                placeholder_text=hint,
                height=42,
                corner_radius=12,
                border_width=2,
                font=ctk.CTkFont(family="Segoe UI", size=12)
            )
            entry.grid(row=1, column=0, sticky="ew", pady=(8, 0))
            self.entries[key] = entry

        self.action_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.action_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        self.action_frame.grid_columnconfigure(0, weight=1)
        self.action_frame.grid_columnconfigure(1, weight=0)
        self.action_frame.grid_columnconfigure(2, weight=0)

        self.predict_btn = ctk.CTkButton(
            self.action_frame,
            text=f"\U0001f33e {self._t('predict_btn')}",
            height=48,
            corner_radius=16,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color=("#1f7a4f", "#159e5a"),
            hover_color=("#1f8b5d", "#1faa6d"),
            command=self.predict
        )
        self.predict_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.autofill_btn = ctk.CTkButton(
            self.action_frame,
            text=f"\U0001f916 {self._t('autofill_btn')}",
            height=48,
            width=120,
            corner_radius=16,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=("#1f4a6f", "#1e5075"),
            hover_color=("#2a6a9e", "#2a6a9e"),
            command=self.open_autofill_dialog
        )
        self.autofill_btn.grid(row=0, column=1, padx=(0, 10))

        self.clear_btn = ctk.CTkButton(
            self.action_frame,
            text=self._t('clear_btn'),
            height=48,
            width=100,
            corner_radius=16,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=("#8b1f2d", "#a5283d"),
            hover_color=("#a42b3f", "#bf3d53"),
            command=self.clear_fields
        )
        self.clear_btn.grid(row=0, column=2)

        self.progress_bar = ctk.CTkProgressBar(self.left_panel, mode="indeterminate")
        self.progress_bar.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.progress_bar.grid_remove()

        self.info_card = ctk.CTkFrame(
            self.left_panel,
            fg_color=("#162022", "#1c292f"),
            corner_radius=18,
            border_width=1,
            border_color=("#224b65", "#1a2d3d")
        )
        self.info_card.grid(row=3, column=0, sticky="ew")
        self.info_card.grid_columnconfigure(0, weight=1)

        self.info_card_label = ctk.CTkLabel(
            self.info_card,
            text=self._t('tip_text'),
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="gray65",
            wraplength=420,
            justify="left"
        )
        self.info_card_label.grid(row=0, column=0, sticky="w", padx=20, pady=18)

        # ========== CONVERSATION (MERGED) PANEL ==========
        self.conversation_card = ctk.CTkFrame(
            self.right_panel,
            fg_color=("#171f23", "#1f2630"),
            corner_radius=20,
            border_width=1,
            border_color=("#2b5065", "#1f3242")
        )
        self.conversation_card.grid(row=0, column=0, sticky="nsew")
        self.conversation_card.grid_columnconfigure(0, weight=1)
        self.conversation_card.grid_rowconfigure(4, weight=1)

        self.conversation_header = ctk.CTkLabel(
            self.conversation_card,
            text=f"\U0001f4ac {self._t('conversation_header')}",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            anchor="w"
        )
        self.conversation_header.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 6))

        # Prediction result
        self.result_label = ctk.CTkLabel(
            self.conversation_card,
            text=f"\U0001f33f {self._t('recommended_crop')}: --",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#d7ffd9",
            wraplength=420,
            justify="left"
        )
        self.result_label.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 2))

        self.confidence_label = ctk.CTkLabel(
            self.conversation_card,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#9bb8c8"
        )
        self.confidence_label.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 6))

        # Top predictions chart
        self.top_predictions_frame = ctk.CTkFrame(
            self.conversation_card, fg_color="transparent"
        )
        self.top_predictions_frame.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 6))
        self.top_predictions_frame.grid_columnconfigure(0, weight=1)
        self.top_predictions_frame.grid_remove()

        self.top_chart_rows = []
        for i in range(5):
            row_f = ctk.CTkFrame(self.top_predictions_frame, fg_color="transparent", height=22)
            row_f.grid(row=i, column=0, sticky="ew", pady=1)
            row_f.grid_columnconfigure(1, weight=1)

            name_lbl = ctk.CTkLabel(
                row_f, text="",
                font=ctk.CTkFont(size=12, weight="bold"),
                width=130, anchor="w"
            )
            name_lbl.grid(row=0, column=0, sticky="w")

            bar = ctk.CTkProgressBar(
                row_f, height=14, corner_radius=7,
                fg_color=("#2a3a3f", "#25333a"),
                progress_color=("#3daf6b", "#45c47a")
            )
            bar.grid(row=0, column=1, sticky="ew", padx=(6, 6))
            bar.set(0)

            pct_lbl = ctk.CTkLabel(
                row_f, text="",
                font=ctk.CTkFont(size=11),
                width=55, anchor="e"
            )
            pct_lbl.grid(row=0, column=2, sticky="e")

            self.top_chart_rows.append((name_lbl, bar, pct_lbl, row_f))

        # Conversation textbox
        self.conversation_text = ctk.CTkTextbox(
            self.conversation_card,
            corner_radius=14,
            border_width=0,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color=("#1d262f", "#222d38"),
            text_color=("#d4dde8", "#e4edf5"),
            wrap="word",
            activate_scrollbars=True
        )
        self.conversation_text.grid(row=4, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self.conversation_text.insert("0.0", self._t('chat_placeholder_empty'))
        self.conversation_text.configure(state="disabled")

        # Chat input
        self.chat_input_frame = ctk.CTkFrame(self.conversation_card, fg_color="transparent")
        self.chat_input_frame.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 18))
        self.chat_input_frame.grid_columnconfigure(0, weight=1)
        self.chat_input_frame.grid_columnconfigure(1, weight=0)

        self.chat_question_entry = ctk.CTkEntry(
            self.chat_input_frame,
            placeholder_text=self._t('chat_placeholder'),
            height=42,
            corner_radius=12,
            border_width=2,
            font=ctk.CTkFont(family="Segoe UI", size=12)
        )
        self.chat_question_entry.grid(row=0, column=0, sticky="ew", pady=(0, 0), padx=(0, 12))
        self.chat_question_entry.bind('<Return>', lambda event: self.ask_follow_up())

        self.chat_send_btn = ctk.CTkButton(
            self.chat_input_frame,
            text=self._t('send_btn'),
            width=110,
            height=42,
            corner_radius=12,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            command=self.ask_follow_up
        )
        self.chat_send_btn.grid(row=0, column=1)

        self.chat_history = []
        self.prediction_context = None

        self.update_llm_status()

    def _t(self, key):
        return STRINGS.get(key, {}).get(self.lang, key)

    def _on_lang_changed(self, choice):
        old_lang = self.lang
        self.lang = "fr" if choice == "FR" else "en"
        if self.lang == old_lang:
            return
        self._apply_language()

    def _apply_language(self):
        self.root.title(f"\U0001f33e {self._t('app_title')}")
        self.title_label.configure(text=f"\U0001f331 {self._t('app_title')}")
        self.subtitle_label.configure(text=self._t('app_subtitle'))
        self.invoke_info.configure(text=self._t('invoke_info'))
        self.input_title.configure(text=f"\U0001f9ea {self._t('input_title')}")
        self.conversation_header.configure(text=f"\U0001f4ac {self._t('conversation_header')}")
        self.result_label.configure(text=f"\U0001f33f {self._t('recommended_crop')}: --")
        self.predict_btn.configure(text=f"\U0001f33e {self._t('predict_btn')}")
        self.clear_btn.configure(text=self._t('clear_btn'))
        self.autofill_btn.configure(text=f"\U0001f916 {self._t('autofill_btn')}")
        self.chat_send_btn.configure(text=self._t('send_btn'))
        self.chat_question_entry.configure(placeholder_text=self._t('chat_placeholder'))
        self.info_card_label.configure(text=self._t('tip_text'))
        self.accuracy_label.configure(
            text=f"\U0001f3af {self._t('model_accuracy')}: {self.accuracy*100:.2f}%"
        )

        # Relabel "Recommended Crop" line
        if self.prediction_context:
            crop = self.prediction_context['crop']
            conf = self.prediction_context.get('confidence', 0)
            self.result_label.configure(text=f"\U0001f33e {self._t('recommended_crop')}: {crop.upper()}")
            self.confidence_label.configure(text=f"{self._t('confidence')}: {conf:.2f}%")

        # Update field labels (don't touch self.entries, just relabel)
        fields = FIELD_LABELS[self.lang]
        for i, (label, key, hint) in enumerate(fields):
            row = (i // 2) + 2
            col = i % 2
            frame = self.input_card.grid_slaves(row=row, column=col)
            if frame:
                frame = frame[0]
                label_widget = frame.grid_slaves(row=0)
                if label_widget:
                    label_widget[0].configure(text=label)
                entry_widget = frame.grid_slaves(row=1)
                if entry_widget:
                    entry_widget[0].configure(placeholder_text=hint)

        self.conversation_text.configure(state="normal")
        placeholder = self._t('chat_placeholder_empty')
        content = self.conversation_text.get("0.0", "end").strip()
        old_placeholders = [STRINGS.get('chat_placeholder_empty', {}).get(l, "") for l in LANGUAGES]
        if content in old_placeholders or not content:
            self.conversation_text.delete("0.0", "end")
            self.conversation_text.insert("0.0", placeholder)
        self.conversation_text.configure(state="disabled")

    def toggle_theme(self):
        current = ctk.get_appearance_mode()
        ctk.set_appearance_mode("light" if current == "Dark" else "dark")

    def _sl(self, key):
        return STRINGS.get(key, {}).get(self.lang, key)

    def validate_input(self, value, field_name, min_val, max_val):
        try:
            num = float(value)
        except ValueError:
            msg = self._sl('invalid_input_msg')
            messagebox.showerror(self._sl('invalid_input'),
                                f"\u274c {field_name} {msg} {min_val} {max_val}!")
            return False, None

        clamped = False
        if num < min_val:
            num = min_val
            clamped = True
        elif num > max_val:
            num = max_val
            clamped = True

        if clamped:
            msg = self._sl('adjusted_input_msg')
            messagebox.showwarning(self._sl('adjusted_input'),
                                   f"\u26a0\ufe0f {field_name} {msg} {min_val} - {max_val}.")

        return True, num

    def append_conversation(self, speaker, message):
        self.conversation_text.configure(state="normal")
        content = self.conversation_text.get("0.0", "end").strip()
        placeholders = [STRINGS.get('chat_placeholder_empty', {}).get(l, "") for l in LANGUAGES]
        if content in placeholders:
            self.conversation_text.delete("0.0", "end")
        labels = {
            "assistant": f"\U0001f916 {self._sl('assistant_label')}",
            "user": f"\U0001f464 {self._sl('you_label')}",
            "system": f"\u2139\ufe0f {self._sl('system_label')}",
        }
        label = labels.get(speaker, speaker.capitalize())
        self.conversation_text.insert("end", f"{label}: {message}\n\n")
        self.conversation_text.see("end")
        self.conversation_text.configure(state="disabled")

    def clear_conversation(self):
        self.chat_history = []
        self.conversation_text.configure(state="normal")
        self.conversation_text.delete("0.0", "end")
        self.conversation_text.insert("0.0", self._t('chat_placeholder_empty'))
        self.conversation_text.configure(state="disabled")

    def render_top_predictions(self, scores):
        for i, (name_lbl, bar, pct_lbl, frame) in enumerate(self.top_chart_rows):
            if i < len(scores):
                crop, conf = scores[i]
                name_lbl.configure(text=f"\U0001f33e {crop.title()}")
                bar.set(float(conf))
                pct_lbl.configure(text=f"{float(conf)*100:.1f}%")
                frame.grid()
            else:
                frame.grid_remove()
        if scores:
            self.top_predictions_frame.grid()
        else:
            self.top_predictions_frame.grid_remove()

    def clear_top_predictions(self):
        for name_lbl, bar, pct_lbl, frame in self.top_chart_rows:
            name_lbl.configure(text="")
            bar.set(0)
            pct_lbl.configure(text="")
            frame.grid_remove()
        self.top_predictions_frame.grid_remove()

    def get_fertilizer_advice(self, crop, input_values):
        cr = load_crop_ranges_json() or {}
        single = cr.get(crop.lower(), {})
        if not single:
            return ""
        nutrient_names = NUTRIENT_NAMES.get(self.lang, NUTRIENT_NAMES["en"])
        keys = ['N', 'P', 'K']
        advice_lines = [f"\u2501\u2501\u2501 {self._sl('fertilizer_header')} \u2501\u2501\u2501"]
        for k, val in zip(keys, input_values[:3]):
            rng = single.get(k)
            if not rng or len(rng) != 2:
                continue
            low, high = rng
            name = nutrient_names.get(k, k)
            if val < low:
                advice_lines.append(
                    f"{name}: {self._sl('fert_add')} {low - val:.0f} kg/ha "
                    f"({self._sl('fert_you_have')} {val:.0f}, "
                    f"{self._sl('fert_need')} {low:.0f}-{high:.0f})"
                )
            elif val > high:
                advice_lines.append(
                    f"{name}: {self._sl('fert_reduce')} {val - high:.0f} kg/ha "
                    f"({self._sl('fert_you_have')} {val:.0f}, "
                    f"{self._sl('fert_excess')})"
                )
            else:
                advice_lines.append(
                    f"{name}: {self._sl('fert_ok')} "
                    f"({val:.0f} {self._sl('fert_need')} {low:.0f}-{high:.0f})"
                )
        return "\n".join(advice_lines)

    def update_llm_status(self):
        if self.llm is not None:
            status_text = f"\u2705 {self._sl('llm_loaded')}"
        else:
            status_text = f"\u274c {self._sl('llm_unavailable')}"
        self.llm_status_label.configure(text=f"LLM: {status_text}")

    def show_loading(self):
        self.progress_bar.grid()
        self.progress_bar.start()
        self.predict_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.autofill_btn.configure(state="disabled")

    def hide_loading(self):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.predict_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        self.autofill_btn.configure(state="normal")

    def fetch_weather(self):
        region = self.region_var.get()
        if region == self._t('all_regions'):
            messagebox.showinfo(self._sl('select_region'), self._sl('select_region_msg'))
            return

        self.weather_btn.configure(state="disabled", text="\u23f3")
        self.root.update()

        def do_fetch():
            result = fetch_weather(region)
            def finish():
                self.weather_btn.configure(state="normal", text="\U0001f324")
                if result is None:
                    messagebox.showerror(self._sl('weather_error'), self._sl('weather_error_msg'))
                    return
                self.entries['temperature'].delete(0, "end")
                self.entries['temperature'].insert(0, f"{result['temperature']:.1f}")
                self.entries['humidity'].delete(0, "end")
                self.entries['humidity'].insert(0, f"{result['humidity']:.0f}")
                self.append_conversation("system",
                    f"{self._sl('weather_autofilled')} {result['city']}: "
                    f"{result['temperature']}\u00b0C, {result['humidity']}%.")
            self.root.after(0, finish)

        threading.Thread(target=do_fetch, daemon=True).start()

    def open_autofill_dialog(self):
        if self.llm is None:
            messagebox.showwarning(self._sl('autofill_btn'), self._sl('autofill_no_llm'))
            return

        dialog = ctk.CTkToplevel(self.root)
        dialog.title(self._sl('autofill_title'))
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ctk.CTkFrame(dialog, fg_color=("#1e2a33", "#1e2a33"))
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        info = ctk.CTkLabel(
            frame, text=self._sl('autofill_prompt'),
            font=ctk.CTkFont(size=12), wraplength=460, justify="left"
        )
        info.grid(row=0, column=0, sticky="w", pady=(0, 12))

        text_entry = ctk.CTkTextbox(
            frame, height=100, corner_radius=10,
            font=ctk.CTkFont(size=12),
            border_width=2, border_color=("#2f546a", "#1d3345")
        )
        text_entry.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        text_entry.insert("0.0", self._sl('autofill_placeholder'))

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="e")
        btn_frame.grid_columnconfigure((0, 1), weight=0)

        fill_btn = ctk.CTkButton(
            btn_frame, text=self._sl('autofill_fill_btn'),
            height=38, corner_radius=10,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=lambda: self._process_autofill(text_entry.get("0.0", "end").strip(), dialog)
        )
        fill_btn.grid(row=0, column=0, padx=(0, 10))

        cancel_btn = ctk.CTkButton(
            btn_frame, text=self._sl('autofill_cancel_btn'),
            height=38, corner_radius=10,
            font=ctk.CTkFont(size=12),
            fg_color=("#5a3a3a", "#4a2a2a"),
            hover_color=("#7a4a4a", "#6a3a3a"),
            command=dialog.destroy
        )
        cancel_btn.grid(row=0, column=1)

        text_entry.focus()

    def _process_autofill(self, description, dialog):
        if not description or description == self._sl('autofill_placeholder'):
            return

        prompt = LLM_AUTOFILL_PROMPTS.get(self.lang, LLM_AUTOFILL_PROMPTS["en"])
        user_msg = f"{prompt}\n\nDescription: {description}"

        dialog.destroy()
        self.show_loading()

        def do_autofill():
            try:
                try:
                    response = self.llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": "You extract field values from descriptions. Respond with JSON only."},
                            {"role": "user", "content": user_msg},
                        ],
                        max_tokens=200,
                        temperature=0.1,
                    )
                    text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                except AttributeError:
                    text = ""
                    try:
                        resp = self.llm(user_msg, max_tokens=200, temperature=0.1, stop=['\n\n'])
                        if isinstance(resp, dict):
                            text = resp.get('choices', [{}])[0].get('text', '') or ''
                        else:
                            text = getattr(resp, 'choices', [{}])[0].get('text', '') or ''
                    except Exception:
                        text = ""

                # Extract JSON from response
                json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(text)

                required = ['N', 'P', 'K', 'organic_carbon', 'temperature', 'humidity', 'ph', 'rainfall']
                ranges = {'N': (0, 70), 'P': (0, 25), 'K': (0, 3000),
                          'organic_carbon': (0, 650),
                          'temperature': (0, 45), 'humidity': (5, 100),
                          'ph': (2.0, 11.0), 'rainfall': (0, 5)}
                missing = []

                def finish_ok():
                    self.hide_loading()
                    for k in required:
                        v = float(data.get(k, 0))
                        lo, hi = ranges[k]
                        v = max(lo, min(hi, v))
                        self.entries[k].delete(0, "end")
                        self.entries[k].insert(0, f"{v:.1f}" if k in ('temperature', 'ph') else f"{v:.0f}")
                    self.append_conversation("system", self._sl('autofill_success'))

                self.root.after(0, finish_ok)

            except Exception as exc:
                print("Autofill error:", exc)
                def finish_err():
                    self.hide_loading()
                    messagebox.showerror(self._sl('autofill_btn'), self._sl('autofill_error'))
                self.root.after(0, finish_err)

        threading.Thread(target=do_autofill, daemon=True).start()

    def build_explanation_messages(self, crop, input_values, confidence):
        keys = ['N', 'P', 'K', 'organic_carbon', 'temperature', 'humidity', 'ph', 'rainfall']
        labels = {
            'N': 'N (kg/ha)', 'P': 'P (kg/ha)', 'K': 'K (kg/ha)',
            'organic_carbon': 'Organic Carbon (g/kg)',
            'temperature': 'Temperature (\u00b0C)', 'humidity': 'Humidity (%)',
            'ph': 'pH', 'rainfall': 'Rainfall (mm)'
        }
        values = dict(zip(keys, input_values))

        crop_ranges = load_crop_ranges_json() or {}
        single = crop_ranges.get(crop.lower(), {})

        range_lines = []
        for k in keys:
            v = single.get(k)
            if v and isinstance(v, (list, tuple)) and len(v) == 2:
                range_lines.append(f"  {labels[k]}: {v[0]:.1f} \u2013 {v[1]:.1f}")
        range_summary = '\n'.join(range_lines) if range_lines else 'unavailable'

        region = self.region_var.get() if self.region_var.get() != self._t('all_regions') else "unspecified"
        sys_prompt = LLM_SYSTEM_PROMPTS.get(self.lang, LLM_SYSTEM_PROMPTS["en"])

        user_prompt = (
            f"The model predicted {crop} with {confidence:.1f}% confidence for region {region}.\n\n"
            f"User's soil & climate values:\n"
            f"N: {values['N']}, P: {values['P']}, K: {values['K']}, "
            f"OC: {values['organic_carbon']}, Temperature: {values['temperature']}\u00b0C, "
            f"Humidity: {values['humidity']}%, pH: {values['ph']}, Rainfall: {values['rainfall']}mm\n\n"
            f"Ideal range for {crop}:\n{range_summary}\n\n"
            f"Answer in 2 sentences:\n"
            f"1. State which values are within or outside the ideal range.\n"
            f"2. Summarize whether {crop} is a good fit.\n"
            f"If confidence is below 60%, mention conditions aren't ideal."
        )

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_explanation(self, crop, input_values, confidence=0.0):
        if self.llm is None:
            explanation = rule_based_explanation(crop, input_values, confidence)
            self.root.after(0, lambda: self.finish_prediction(explanation))
            return

        messages = self.build_explanation_messages(crop, input_values, confidence)
        try:
            try:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=120,
                    temperature=0.35,
                )
            except AttributeError:
                response = self.llm(prompt=messages[-1]["content"], max_tokens=120, temperature=0.35, stop=['\n'])
                explanation = ""
                if isinstance(response, dict):
                    choices = response.get('choices', [])
                    if choices:
                        explanation = choices[0].get('text', '') or ''
                else:
                    choices = getattr(response, 'choices', [])
                    if choices:
                        first = choices[0]
                        explanation = getattr(first, 'text', '') or (first.get('text', '') if isinstance(first, dict) else '')
                explanation = (explanation or '').strip()
                if explanation:
                    explanation = re.sub(r'\s{2,}', ' ', explanation).strip()
                    if not re.search(r'[.!?]$', explanation):
                        explanation += '.'
                else:
                    explanation = rule_based_explanation(crop, input_values, confidence)
                def finish():
                    self.finish_prediction(explanation)
                self.root.after(0, finish)
                return

            explanation = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            explanation = (explanation or '').strip()
            if explanation:
                explanation = re.sub(r'\s{2,}', ' ', explanation).strip()
                if not re.search(r'[.!?]$', explanation):
                    explanation += '.'
            else:
                explanation = rule_based_explanation(crop, input_values, confidence)
        except Exception as exc:
            explanation = rule_based_explanation(crop, input_values, confidence)
            print("LLM error, using fallback:", exc)

        self.root.after(0, lambda: self.finish_prediction(explanation))

    def finish_prediction(self, explanation):
        self.hide_loading()
        if self.prediction_context is None:
            return
        ctx = self.prediction_context

        # Append explanation
        self.append_conversation("assistant", explanation)

        # Append fertilizer advice
        fert = self.get_fertilizer_advice(ctx['crop'], ctx['values'])
        if fert:
            self.append_conversation("system", fert)

        self.chat_history = [("assistant", explanation)]

        # Render top 5 chart
        if 'top_scores' in ctx:
            self.render_top_predictions(ctx['top_scores'])

    def build_chat_messages(self, question):
        if self.prediction_context is None:
            raise ValueError("No prediction context available for chat responses.")

        ctx = self.prediction_context
        input_values = ctx['values']
        values = dict(zip(['N', 'P', 'K', 'organic_carbon', 'temperature', 'humidity', 'ph', 'rainfall'], input_values))
        confidence = ctx.get('confidence', 0)

        context_msg = (
            f"Predicted crop: {ctx['crop']} (confidence: {confidence:.1f}%)\n"
            f"User's inputs: N={values['N']}, P={values['P']}, K={values['K']}, "
            f"OC={values['organic_carbon']}, Temp={values['temperature']}\u00b0C, "
            f"Humidity={values['humidity']}%, pH={values['ph']}, Rainfall={values['rainfall']}mm\n"
        )

        sys_prompt = LLM_SYSTEM_PROMPTS.get(self.lang, LLM_SYSTEM_PROMPTS["en"])
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "assistant", "content": context_msg},
        ]
        for role, text in self.chat_history[-6:]:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": question})
        return messages

    def generate_chat_response(self, question):
        if self.llm is None:
            self.root.after(0, lambda: self.append_conversation(
                "assistant", self._sl('llm_chat_unavailable')))
            return

        try:
            messages = self.build_chat_messages(question)
            try:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=200,
                    temperature=0.35,
                )
            except AttributeError:
                prompt = messages[-1]["content"]
                try:
                    response = self.llm(prompt, max_tokens=200, temperature=0.35, stop=['\n\n'])
                except TypeError:
                    response = self.llm.create(prompt=prompt, max_tokens=200, temperature=0.35, stop=['\n\n'])
                explanation = ""
                if isinstance(response, dict):
                    choices = response.get('choices', [])
                    if choices:
                        explanation = choices[0].get('text', '') or ''
                else:
                    choices = getattr(response, 'choices', [])
                    if choices:
                        first = choices[0]
                        explanation = getattr(first, 'text', '') or (first.get('text', '') if isinstance(first, dict) else '')
                explanation = (explanation or '').strip()
                explanation = re.sub(r'\s+', ' ', explanation).strip()
                if not explanation:
                    explanation = "I couldn't produce a clear answer. Try rephrasing."
                if not re.search(r'[.!?]$', explanation):
                    explanation += '.'
                self.chat_history.append(("assistant", explanation))
                self.root.after(0, lambda: self.append_conversation("assistant", explanation))
                return

            explanation = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            explanation = (explanation or '').strip()
            explanation = re.sub(r'\s+', ' ', explanation).strip()
            if not explanation:
                explanation = "I couldn't produce a clear answer. Try rephrasing."
            if not re.search(r'[.!?]$', explanation):
                explanation += '.'

            self.chat_history.append(("assistant", explanation))
            self.root.after(0, lambda: self.append_conversation("assistant", explanation))
        except Exception as exc:
            print("LLM chat error:", exc)
            self.root.after(0, lambda: self.append_conversation(
                "assistant", f"Chat response failed: {str(exc)}"))

    def ask_follow_up(self):
        question = self.chat_question_entry.get().strip()
        if not question:
            return

        if self.prediction_context is None:
            messagebox.showwarning(self._sl('no_prediction'), self._sl('no_prediction_msg'))
            return

        self.chat_question_entry.delete(0, "end")
        self.chat_history.append(("user", question))
        self.append_conversation("user", question)
        threading.Thread(target=self.generate_chat_response, args=(question,), daemon=True).start()

    def get_region_filtered_crops(self):
        selected = self.region_var.get()
        if selected == self._t('all_regions') or self.crop_region_map is None:
            return None
        filtered = [c for c, regions in self.crop_region_map.items() if selected in regions]
        return filtered

    def predict(self):
        ranges = {
            'N': (0, 55), 'P': (5, 25), 'K': (2, 2800),
            'organic_carbon': (0, 650),
            'temperature': (-5, 40), 'humidity': (5, 100),
            'ph': (2.0, 11.0), 'rainfall': (0, 16)
        }

        input_values = []
        features_order = ['N', 'P', 'K', 'organic_carbon', 'temperature', 'humidity', 'ph', 'rainfall']

        for feature in features_order:
            value = self.entries[feature].get().strip()

            if value == "":
                messagebox.showwarning(self._sl('missing_input'),
                                       f"\u26a0\ufe0f {self._sl('missing_input_msg')} {feature}!")
                return

            min_val, max_val = ranges[feature]
            field_name = feature.capitalize()
            if feature == 'temperature':
                field_name = 'Temperature (\u00b0C)'
            elif feature == 'humidity':
                field_name = 'Humidity (%)'
            elif feature == 'rainfall':
                field_name = 'Rainfall (mm)'
            elif feature == 'ph':
                field_name = 'pH'
            elif feature == 'organic_carbon':
                field_name = 'Organic Carbon (g/kg)'

            valid, num = self.validate_input(value, field_name, min_val, max_val)
            if not valid:
                return

            input_values.append(num)

        try:
            features = input_values[:]
            if self.continent_encoder is not None:
                continent = self.continent_var.get()
                cont_encoded = self.continent_encoder.transform([continent])[0]
                features.append(cont_encoded)
            input_array = np.array(features).reshape(1, -1)

            probabilities = self.model.predict_proba(input_array)[0]
            crop_indices = list(range(len(probabilities)))
            crop_scores = [(self.label_encoder.inverse_transform([i])[0], probabilities[i]) for i in crop_indices]
            crop_scores.sort(key=lambda x: -x[1])

            region_filter = self.get_region_filtered_crops()
            if region_filter:
                crop_scores = [(c, s) for c, s in crop_scores if c in region_filter]
                if not crop_scores:
                    crop_scores = [(self.label_encoder.inverse_transform([i])[0], probabilities[i]) for i in crop_indices]

            predicted_crop = crop_scores[0][0]
            confidence = crop_scores[0][1] * 100

            self.result_label.configure(text=f"\U0001f33e {self._t('recommended_crop')}: {predicted_crop.upper()}")
            self.confidence_label.configure(text=f"{self._t('confidence')}: {confidence:.2f}%")

            self.prediction_context = {
                'crop': predicted_crop,
                'values': input_values,
                'confidence': confidence,
                'top_scores': crop_scores[:5],
            }
            self.clear_conversation()
            self.clear_top_predictions()

            if confidence > 80:
                self.result_label.configure(text_color=("green4", "#4ade80"))
            elif confidence > 60:
                self.result_label.configure(text_color=("orange3", "#fb923c"))
            else:
                self.result_label.configure(text_color=("red3", "#f87171"))

            self.show_loading()
            threading.Thread(
                target=self.generate_explanation,
                args=(predicted_crop, input_values, confidence),
                daemon=True
            ).start()

        except Exception as e:
            self.hide_loading()
            self.clear_conversation()
            self.clear_top_predictions()
            messagebox.showerror(self._sl('prediction_error'),
                                f"{self._sl('prediction_error_msg')}:\n{str(e)}")

    def on_soil_type_changed(self, choice):
        if choice == self._t('manual_entry') or self.soil_profiles is None:
            return
        profile = self.soil_profiles.get(choice)
        if profile is None:
            return
        self.entries['N'].delete(0, "end")
        self.entries['N'].insert(0, str(profile['N_median']))
        self.entries['P'].delete(0, "end")
        self.entries['P'].insert(0, str(profile['P_median']))
        self.entries['K'].delete(0, "end")
        self.entries['K'].insert(0, str(profile['K_median']))
        self.entries['ph'].delete(0, "end")
        self.entries['ph'].insert(0, str(profile['ph_median']))
        if 'organic_carbon_median' in profile:
            self.entries['organic_carbon'].delete(0, "end")
            self.entries['organic_carbon'].insert(0, str(profile['organic_carbon_median']))

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, "end")
        self.soil_type_var.set(self._t('manual_entry'))
        self.result_label.configure(text=f"\U0001f33f {self._t('recommended_crop')}: --",
                                    text_color=("gray40", "gray70"))
        self.confidence_label.configure(text="")
        self.clear_conversation()
        self.clear_top_predictions()
        self.prediction_context = None


# ============ MAIN EXECUTION ============
def main():
    print("\U0001f33e Initializing Crop Recommendation System...")

    model, label_encoder, continent_encoder, accuracy = train_model()

    if model is None:
        print("Failed to load model. Please check your dataset and model files.")
        return

    print(f"\u2705 Model loaded successfully! Accuracy: {accuracy*100:.2f}%")
    print(f"\U0001f4ca Number of crops: {len(label_encoder.classes_)}")

    print("\u23f3 Loading LLM model...")
    llm = load_llm_model()
    if llm is None:
        print("\u26a0\ufe0f LLM is not available. Explanations will use rule-based fallback.")

    print("\U0001f680 Launching GUI...")

    root = ctk.CTk()
    app = CropRecommendationApp(root, model, label_encoder, continent_encoder, accuracy, llm=llm)
    root.mainloop()


if __name__ == "__main__":
    main()
