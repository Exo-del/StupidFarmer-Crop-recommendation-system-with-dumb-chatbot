import json
import os
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
from tkinter import messagebox
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

# ============ CONFIGURATION ============
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

LLM_MODEL_PATH = BASE_DIR / 'LLM' / 'LittleLamb-ToolCalling.f16.gguf'
CROP_RANGES_PATH = BASE_DIR / 'crop_ranges.json'
LLM_CONTEXT_WINDOW = 8000
LLM_PROMPT_TEMPLATE = """You are a warm, concise agricultural assistant. Use the crop range summary below to explain why the predicted crop is recommended for these soil and climate values, and include one small fun fact about the result.

Start the answer with "Did you know" and keep it informative, friendly, and one paragraph or less.
Do not include any reasoning, thought process, or extra explanation.
Use all of the following values: N, P, K, temperature, humidity, pH, and rainfall.
Do not think aloud; answer directly in a single final paragraph,if the user asks with unrelated questions anwser shortly and effectively.

Crop ranges:
{crop_ranges}

Current input values:
N: {N}
P: {P}
K: {K}
Temperature: {temperature} °C
Humidity: {humidity} %
pH: {ph}
Rainfall: {rainfall} mm

Predicted crop: {crop}

Answer:"""

LLM_CHAT_PROMPT_TEMPLATE = """You are a warm, concise agricultural assistant. The user has already received a recommended crop and the input conditions used to make that recommendation.
Use the crop range summary below as context, then answer the follow-up question directly and politely.
Keep the response informative and friendly, one paragraph or less. Do not include your internal reasoning.
Do not output tags like <think> or phrases like Thinking Process.
Answer directly with a final assistant response only.

Crop ranges:
{crop_ranges}

Current input values:
N: {N}
P: {P}
K: {K}
Temperature: {temperature} °C
Humidity: {humidity} %
pH: {ph}
Rainfall: {rainfall} mm

Predicted crop: {crop}

Conversation history:
{history}

User question: {question}

Assistant:"""

# ============ LOAD THE SAVED MODEL ============
def train_model():
    """Load the saved XGBoost model and prepare the label encoder"""
    try:
        df = pd.read_csv(BASE_DIR / 'dataset' / 'Crop_recommendation.csv')
    except FileNotFoundError:
        messagebox.showerror("Error", f"Dataset file not found!\nMake sure '{BASE_DIR / 'dataset' / 'Crop_recommendation.csv'}' exists.")
        return None, None, None

    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features]
    y = df['label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    try:
        model = joblib.load('xgboost_model.pkl')
    except FileNotFoundError:
        messagebox.showerror("Error", "Saved model file not found!\nMake sure 'xgboost_model.pkl' exists in the application folder.")
        return None, None, None
    except Exception as exc:
        messagebox.showerror("Error", f"Failed to load saved model:\n{str(exc)}")
        return None, None, None

    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    accuracy = model.score(X_test, y_test)

    return model, le, accuracy


def load_llm_model():
    print(f"LLM model path: {LLM_MODEL_PATH}")
    if Llama is None:
        print("⚠️ llama_cpp is not installed. LLM explanations are disabled.")
        return None

    if not LLM_MODEL_PATH.exists():
        print(f"⚠️ LLM model file not found at: {LLM_MODEL_PATH}")
        return None

    try:
        llm = Llama(
            model_path=str(LLM_MODEL_PATH),
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
                f"K {values['K'][0]:.0f}-{values['K'][1]:.0f}, "
                f"Temp {values['temperature'][0]:.1f}-{values['temperature'][1]:.1f}, "
                f"Humidity {values['humidity'][0]:.1f}-{values['humidity'][1]:.1f}, "
                f"pH {values['ph'][0]:.2f}-{values['ph'][1]:.2f}, "
                f"Rain {values['rainfall'][0]:.1f}-{values['rainfall'][1]:.1f}"
            )
            lines.append(line)
        return "\n".join(lines)

    try:
        df = pd.read_csv('dataset/Crop_recommendation.csv')
    except FileNotFoundError:
        return "Crop range data unavailable."

    lines = []
    for crop in sorted(df['label'].unique()):
        crop_df = df[df['label'] == crop]
        line = (
            f"{crop}: N {crop_df['N'].min():.0f}-{crop_df['N'].max():.0f}, "
            f"P {crop_df['P'].min():.0f}-{crop_df['P'].max():.0f}, "
            f"K {crop_df['K'].min():.0f}-{crop_df['K'].max():.0f}, "
            f"Temp {crop_df['temperature'].min():.1f}-{crop_df['temperature'].max():.1f}, "
            f"Humidity {crop_df['humidity'].min():.1f}-{crop_df['humidity'].max():.1f}, "
            f"pH {crop_df['ph'].min():.2f}-{crop_df['ph'].max():.2f}, "
            f"Rain {crop_df['rainfall'].min():.1f}-{crop_df['rainfall'].max():.1f}"
        )
        lines.append(line)

    return "\n".join(lines)


# ============ GUI APPLICATION ============
class CropRecommendationApp:
    def __init__(self, root, model, label_encoder, accuracy, llm=None):
        self.root = root
        self.model = model
        self.label_encoder = label_encoder
        self.accuracy = accuracy
        self.llm = llm
        self.crop_ranges_summary = build_crop_range_summary()

        self.root.title("🌾 Crop Recommendation System")
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
            text="🌱 Crop Recommendation System",
            font=ctk.CTkFont(family="Segoe UI", size=30, weight="bold"),
            text_color=("#d7ffd9", "#d7ffd9")
        )
        self.title_label.grid(row=0, column=0, sticky="w", padx=24, pady=(24, 6))

        self.subtitle_label = ctk.CTkLabel(
            self.hero_card,
            text="AI-powered soil and climate analysis for smarter crop selection, presented in a clean, modern dashboard.",
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
            text=f"🎯 Model Accuracy: {accuracy*100:.2f}%",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#98ffb5"
        )
        self.accuracy_label.pack(padx=18, pady=14)

        self.llm_status_badge = ctk.CTkFrame(
            self.status_row,
            fg_color=("#22344f", "#1f2d43"),
            corner_radius=18,
            border_width=1,
            border_color=("#3a5a9e", "#1b2a3f")
        )
        self.llm_status_badge.grid(row=0, column=1, sticky="e", padx=(14, 0))
        self.llm_status_label = ctk.CTkLabel(
            self.llm_status_badge,
            text="LLM status: checking...",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#b9d2ff"
        )
        self.llm_status_label.pack(padx=18, pady=14)

        self.invoke_info = ctk.CTkLabel(
            self.hero_card,
            text="Enter values on the left, then get a crop recommendation with explanation and follow-up insights.",
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
        self.right_panel.grid_rowconfigure(2, weight=1)

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
            text="🧪 Soil & Climate Inputs",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold")
        )
        self.input_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=22, pady=(22, 12))

        self.entries = {}
        fields = [
            ('Nitrogen (N)', 'N', '0 – 140 kg/ha'),
            ('Phosphorus (P)', 'P', '0 – 100 kg/ha'),
            ('Potassium (K)', 'K', '0 – 200 kg/ha'),
            ('Temperature', 'temperature', '10 – 40 °C'),
            ('Humidity', 'humidity', '15 – 100 %'),
            ('pH Level', 'ph', '4.5 – 8.5'),
            ('Rainfall', 'rainfall', '20 – 300 mm')
        ]

        for i, (label, key, hint) in enumerate(fields):
            row = (i // 2) + 1
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

        self.predict_btn = ctk.CTkButton(
            self.action_frame,
            text="🌾 Predict Crop",
            height=48,
            corner_radius=16,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color=("#1f7a4f", "#159e5a"),
            hover_color=("#1f8b5d", "#1faa6d"),
            command=self.predict
        )
        self.predict_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.clear_btn = ctk.CTkButton(
            self.action_frame,
            text="Clear All",
            height=48,
            width=128,
            corner_radius=16,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=("#8b1f2d", "#a5283d"),
            hover_color=("#a42b3f", "#bf3d53"),
            command=self.clear_fields
        )
        self.clear_btn.grid(row=0, column=1)

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
            text="Tip: Use accurate soil data and rainfall values for the best crop match. Follow-up questions can help refine recommendations.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="gray65",
            wraplength=420,
            justify="left"
        )
        self.info_card_label.grid(row=0, column=0, sticky="w", padx=20, pady=18)

        # ========== OUTPUT PANELS ==========
        self.result_card = ctk.CTkFrame(
            self.right_panel,
            fg_color=("#171f23", "#1f2630"),
            corner_radius=20,
            border_width=1,
            border_color=("#2b5065", "#1f3242")
        )
        self.result_card.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        self.result_card.grid_columnconfigure(0, weight=1)

        self.result_label = ctk.CTkLabel(
            self.result_card,
            text="🌿 Recommended Crop: --",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#d7ffd9",
            wraplength=420,
            justify="left"
        )
        self.result_label.grid(row=0, column=0, sticky="w", padx=20, pady=(22, 6))

        self.confidence_label = ctk.CTkLabel(
            self.result_card,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#9bb8c8"
        )
        self.confidence_label.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 20))

        self.explanation_card = ctk.CTkFrame(
            self.right_panel,
            fg_color=("#171f23", "#1f2630"),
            corner_radius=20,
            border_width=1,
            border_color=("#2b5065", "#1f3242")
        )
        self.explanation_card.grid(row=1, column=0, sticky="nsew", pady=(0, 18))
        self.explanation_card.grid_columnconfigure(0, weight=1)
        self.explanation_card.grid_rowconfigure(1, weight=1)

        self.explanation_header = ctk.CTkLabel(
            self.explanation_card,
            text="💡 Why this crop?",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            anchor="w"
        )
        self.explanation_header.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        self.explanation_text = ctk.CTkTextbox(
            self.explanation_card,
            height=180,
            corner_radius=14,
            border_width=0,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color=("#1d262f", "#222d38"),
            text_color=("#d4dde8", "#e4edf5"),
            wrap="word",
            activate_scrollbars=True
        )
        self.explanation_text.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.explanation_text.insert("0.0", "Enter soil parameters and click Predict to see the AI explanation.")
        self.explanation_text.configure(state="disabled")

        self.chat_card = ctk.CTkFrame(
            self.right_panel,
            fg_color=("#171f23", "#1f2630"),
            corner_radius=20,
            border_width=1,
            border_color=("#2b5065", "#1f3242")
        )
        self.chat_card.grid(row=2, column=0, sticky="nsew")
        self.chat_card.grid_columnconfigure(0, weight=1)
        self.chat_card.grid_rowconfigure(1, weight=1)

        self.chat_header = ctk.CTkLabel(
            self.chat_card,
            text="💬 Follow-up Chat",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            anchor="w"
        )
        self.chat_header.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        self.chat_history_text = ctk.CTkTextbox(
            self.chat_card,
            height=220,
            corner_radius=14,
            border_width=0,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color=("#1d262f", "#222d38"),
            text_color=("#d4dde8", "#e4edf5"),
            wrap="word",
            activate_scrollbars=True
        )
        self.chat_history_text.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self.chat_history_text.insert("0.0", "Ask a follow-up question once a crop prediction is ready.")
        self.chat_history_text.configure(state="disabled")

        self.chat_input_frame = ctk.CTkFrame(self.chat_card, fg_color="transparent")
        self.chat_input_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        self.chat_input_frame.grid_columnconfigure(0, weight=1)
        self.chat_input_frame.grid_columnconfigure(1, weight=0)

        self.chat_question_entry = ctk.CTkEntry(
            self.chat_input_frame,
            placeholder_text="Ask a follow-up question...",
            height=42,
            corner_radius=12,
            border_width=2,
            font=ctk.CTkFont(family="Segoe UI", size=12)
        )
        self.chat_question_entry.grid(row=0, column=0, sticky="ew", pady=(0, 0), padx=(0, 12))
        self.chat_question_entry.bind('<Return>', lambda event: self.ask_follow_up())

        self.chat_send_btn = ctk.CTkButton(
            self.chat_input_frame,
            text="Send",
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

    def toggle_theme(self):
        current = ctk.get_appearance_mode()
        ctk.set_appearance_mode("light" if current == "Dark" else "dark")

    def validate_input(self, value, field_name, min_val, max_val):
        try:
            num = float(value)
        except ValueError:
            messagebox.showerror("Invalid Input",
                                f"❌ {field_name} must be a number between {min_val} and {max_val}!\n"
                                f"Please enter a valid numeric value.")
            return False, None

        clamped = False
        if num < min_val:
            num = min_val
            clamped = True
        elif num > max_val:
            num = max_val
            clamped = True

        if clamped:
            messagebox.showwarning(
                "Adjusted Input",
                f"⚠️ {field_name} was adjusted to the safe range {min_val} to {max_val}."
            )

        return True, num

    def update_explanation_text(self, text):
        self.explanation_text.configure(state="normal")
        self.explanation_text.delete("0.0", "end")
        self.explanation_text.insert("0.0", text)
        self.explanation_text.configure(state="disabled")

    def append_chat_message(self, speaker, message):
        self.chat_history_text.configure(state="normal")
        self.chat_history_text.insert("end", f"{speaker}: {message}\n\n")
        self.chat_history_text.see("end")
        self.chat_history_text.configure(state="disabled")

    def clear_chat_history(self):
        self.chat_history = []
        self.chat_history_text.configure(state="normal")
        self.chat_history_text.delete("0.0", "end")
        self.chat_history_text.insert("0.0", "Ask a follow-up question once a crop prediction is ready.")
        self.chat_history_text.configure(state="disabled")

    def update_llm_status(self):
        status_text = "✅ LLM loaded" if self.llm is not None else "❌ LLM unavailable"
        self.llm_status_label.configure(text=f"LLM status: {status_text}")

    def show_loading(self):
        self.progress_bar.grid()
        self.progress_bar.start()
        self.predict_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")

    def hide_loading(self):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.predict_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")

    def build_explanation_prompt(self, crop, input_values):
        values = dict(zip(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], input_values))

        crop_ranges = load_crop_ranges_json() or {}
        single = crop_ranges.get(crop.lower(), {})

        parts = []
        for k in ('N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'):
            v = single.get(k)
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 2:
                parts.append(f"{k}:{v[0]}-{v[1]}")
            else:
                parts.append(f"{k}:{v}")

        compact_ranges = ', '.join(parts) if parts else 'ranges unavailable'

        return LLM_PROMPT_TEMPLATE.format(
            crop_ranges=compact_ranges,
            crop=crop,
            **values
        )

    def generate_explanation(self, crop, input_values):
        if self.llm is None:
            self.root.after(0, lambda: (self.update_explanation_text(
                "LLM explanation unavailable. Install llama-cpp-python and place the GGUF model in the LLM folder."), self.hide_loading()))
            return

        def rule_based_explanation(crop, input_values):
            cr = load_crop_ranges_json() or {}
            single = cr.get(crop.lower(), {})
            keys = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            labels = {
                'N': 'nitrogen', 'P': 'phosphorus', 'K': 'potassium',
                'temperature': 'temperature', 'humidity': 'humidity',
                'ph': 'pH', 'rainfall': 'rainfall'
            }

            environment_facts = {
                'rice': 'it thrives in warm, moist fields with steady water availability',
                'maize': 'it prefers warm, sunny weather and well-drained soil with moderate rainfall',
                'chickpea': 'it does well in drier, well-drained fields with moderate moisture',
                'kidneybeans': 'it grows best in warm soil and needs consistent moisture during flowering',
                'pigeonpeas': 'it likes warm climates and can tolerate dry spells once established',
                'mothbeans': 'it is suited to hot, semi-arid conditions and low rainfall',
                'mungbean': 'it thrives in warm soils with moderate humidity and good drainage',
                'blackgram': 'it favors warm, humid weather and well-drained soils',
                'lentil': 'it prefers cool, dry conditions and well-drained sandy soil',
                'pomegranate': 'it grows well in warm, dry climates with deep, well-drained soil',
                'banana': 'it loves warm, humid conditions with plenty of water and rich soil',
                'mango': 'it flourishes in warm, frost-free climates with good drainage',
                'grapes': 'they prefer sunny, dry weather and well-drained soil for healthy vines',
                'watermelon': 'it likes hot weather, plenty of sun, and consistent moisture',
                'muskmelon': 'it does well in warm, sunny fields with good drainage',
                'apple': 'it grows best in cooler climates with well-drained, fertile soil',
                'orange': 'it prefers warm, subtropical weather and well-drained soil',
                'papaya': 'it thrives in warm, humid conditions with rich, well-drained soil',
                'coconut': 'it flourishes in tropical, coastal conditions with ample moisture',
                'cotton': 'it prefers warm temperatures and well-drained, fertile soils',
                'jute': 'it grows best in warm, humid regions with plenty of moisture',
                'coffee': 'it favors warm, shaded areas with rich, well-drained soil'
            }

            # Safe allowed bounds for each feature to avoid nonsensical ranges
            allowed_bounds = {
                'N': (0, 10000),
                'P': (0, 10000),
                'K': (0, 10000),
                'temperature': (-50, 100),
                'humidity': (0, 100),
                'ph': (0.0, 14.0),
                'rainfall': (0, 10000)
            }

            def fmt_range(low, high, key):
                if key in ['temperature', 'humidity', 'rainfall']:
                    return f"{low:.1f}-{high:.1f}"
                if key == 'ph':
                    return f"{low:.2f}-{high:.2f}"
                return f"{int(low)}-{int(high)}"

            matched = []
            for k, val in zip(keys, input_values):
                rng = single.get(k)
                if not rng or len(rng) != 2:
                    continue
                try:
                    low, high = float(rng[0]), float(rng[1])
                except Exception:
                    continue

                # Clamp crop's stored ranges to allowed bounds and skip invalid ranges
                a_min, a_max = allowed_bounds.get(k, (None, None))
                if a_min is not None and a_max is not None:
                    low = max(low, a_min)
                    high = min(high, a_max)

                if low > high:
                    continue
                if low == 0 and high == 0:
                    continue

                try:
                    v = float(val)
                except Exception:
                    continue

                if low <= v <= high:
                    matched.append((k, low, high))

            fact = environment_facts.get(crop.lower(), 'it also prefers a development environment that matches these conditions')

            if len(matched) == len(keys):
                ranges = [f"{labels[k]} {fmt_range(low, high, k)}" for k, low, high in matched]
                return f"Did you know {crop} is recommended because {', '.join(ranges[:-1])}, and {ranges[-1]} all fall within ideal ranges, and {fact}."

            if matched:
                ranges = [f"{labels[k]} {fmt_range(low, high, k)}" for k, low, high in matched[:3]]
                if len(ranges) == 1:
                    return f"Did you know {crop} is recommended because {ranges[0]} is within ideal range, and {fact}."
                if len(ranges) == 2:
                    return f"Did you know {crop} is recommended because {ranges[0]} and {ranges[1]} are within ideal range, and {fact}."
                return f"Did you know {crop} is recommended because {', '.join(ranges[:-1])}, and {ranges[-1]} are within ideal range, and {fact}."

            return f"Did you know {crop} is recommended for these soil and climate conditions, and {fact}."

        prompt = self.build_explanation_prompt(crop, input_values)
        try:
            try:
                response = self.llm(prompt, max_tokens=120, temperature=0.35, stop=['\n'])
            except TypeError:
                response = self.llm.create(prompt=prompt, max_tokens=120, temperature=0.35, stop=['\n'])

            print("LLM response raw:", response)
            explanation = ""
            if isinstance(response, dict):
                choices = response.get('choices', [])
                if choices:
                    explanation = choices[0].get('text', '') or getattr(choices[0], 'text', '')
            else:
                choices = getattr(response, 'choices', [])
                if choices:
                    first = choices[0]
                    explanation = getattr(first, 'text', '') or first.get('text', '') if isinstance(first, dict) else ''

            explanation = (explanation or '').strip()

            if explanation:
                lower = explanation.lower()
                if 'answer:' in lower:
                    if 'Answer:' in explanation:
                        explanation = explanation.split('Answer:', 1)[1].strip()
                    else:
                        explanation = explanation.split('answer:', 1)[1].strip()

            explanation = re.sub(r'(?is)^.*?\banswer:\s*', '', explanation).strip()
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', explanation) if p.strip()]
            chosen = paragraphs[0] if paragraphs else explanation
            chosen = chosen.replace('\n', ' ').strip()

            if chosen:
                chosen = re.sub(r"^\s*Use the[^\.\n]*(?:[\.\n]|$)", "", chosen, flags=re.IGNORECASE).strip()
                chosen = re.sub(r"^\s*(okay,?\s*(let's|lets)\s*break\s*this\s*down[\.\:\s]*)", "", chosen, flags=re.IGNORECASE).strip()
                chosen = re.sub(r"^\s*(sure,?\s*)", "", chosen, flags=re.IGNORECASE).strip()
                chosen = re.sub(r"^\s*(the user wants me|i should|i need to|the inputs are|the crop's ideal|the crop is|the answer is)[^\.\n]*(?:[\.\n]|$)", "", chosen, flags=re.IGNORECASE).strip()
                chosen = re.sub(r'\[.*?\]', '', chosen).strip()
                chosen = re.sub(r'\b\d+\.|\b\d+\)', '', chosen).strip()
                chosen = re.sub(r'\s{2,}', ' ', chosen)

                if not re.search(r'[.!?]$', chosen):
                    chosen += '.'

                if 'did you know' not in chosen.lower():
                    if 'did you know' in chosen.lower():
                        idx = chosen.lower().index('did you know')
                        chosen = chosen[idx:].strip()
                    else:
                        chosen = f"Did you know {crop} is recommended because these soil and climate values match its ideal growing environment?"

                explanation = chosen
            else:
                explanation = rule_based_explanation(crop, input_values)
        except Exception as exc:
            explanation = f"Explanation generation failed: {str(exc)}"
            print("LLM error:", exc)

        def finish():
            self.update_explanation_text(explanation)
            self.hide_loading()
            if self.prediction_context:
                self.chat_history = [("assistant", explanation)]
                self.chat_history_text.configure(state="normal")
                self.chat_history_text.delete("0.0", "end")
                self.chat_history_text.configure(state="disabled")
                self.append_chat_message("Assistant", explanation)

        self.root.after(0, finish)

    def build_chat_prompt(self, question):
        if self.prediction_context is None:
            raise ValueError("No prediction context available for chat responses.")

        input_values = self.prediction_context['values']
        values = dict(zip(['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], input_values))
        history = '\n'.join([f"{role.title()}: {text}" for role, text in self.chat_history])

        return LLM_CHAT_PROMPT_TEMPLATE.format(
            crop_ranges=self.crop_ranges_summary,
            crop=self.prediction_context['crop'],
            history=history or 'None',
            question=question,
            **values
        )

    def generate_chat_response(self, question):
        if self.llm is None:
            self.root.after(0, lambda: self.append_chat_message(
                "Assistant",
                "LLM chat unavailable. Install llama-cpp-python and place the GGUF model in the LLM folder."))
            return

        try:
            prompt = self.build_chat_prompt(question)
            try:
                response = self.llm(prompt, max_tokens=150, temperature=0.35, stop=['\n\n', '\nAssistant:', '\nUser:', 'Answer:'])
            except TypeError:
                response = self.llm.create(prompt=prompt, max_tokens=150, temperature=0.35, stop=['\n\n', '\nAssistant:', '\nUser:', 'Answer:'])

            explanation = ""
            if isinstance(response, dict):
                choices = response.get('choices', [])
                if choices:
                    explanation = choices[0].get('text', '') or getattr(choices[0], 'text', '')
            else:
                choices = getattr(response, 'choices', [])
                if choices:
                    first = choices[0]
                    explanation = getattr(first, 'text', '') or first.get('text', '') if isinstance(first, dict) else ''

            explanation = (explanation or '').strip()
            explanation = re.sub(r'(?is)^.*?\banswer:\s*', '', explanation).strip()
            explanation = re.sub(r'(?is)<think>.*?</think>', '', explanation).strip()
            explanation = re.sub(r'(?i)thinking process:?', '', explanation).strip()
            explanation = re.sub(r'(?i)assistant:?', '', explanation).strip()
            explanation = re.sub(r'(?i)user:?', '', explanation).strip()
            explanation = re.sub(r'\s+', ' ', explanation).strip()
            if not explanation:
                explanation = "Sorry, I could not produce a clear follow-up answer."
            if not re.search(r'[.!?]$', explanation):
                explanation += '.'

            self.chat_history.append(("assistant", explanation))
            self.root.after(0, lambda: self.append_chat_message("Assistant", explanation))
        except Exception as exc:
            print("LLM chat error:", exc)
            self.root.after(0, lambda: self.append_chat_message(
                "Assistant",
                f"Chat response failed: {str(exc)}"))

    def ask_follow_up(self):
        question = self.chat_question_entry.get().strip()
        if not question:
            return

        if self.prediction_context is None:
            messagebox.showwarning("No Prediction", "Please generate a crop prediction before asking follow-up questions.")
            return

        self.chat_question_entry.delete(0, "end")
        self.chat_history.append(("user", question))
        self.append_chat_message("You", question)
        threading.Thread(target=self.generate_chat_response, args=(question,), daemon=True).start()

    def predict(self):
        ranges = {
            'N': (0, 140),
            'P': (0, 100),
            'K': (0, 200),
            'temperature': (10, 40),
            'humidity': (15, 100),
            'ph': (4.5, 8.5),
            'rainfall': (20, 300)
        }

        input_values = []
        features_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

        for feature in features_order:
            value = self.entries[feature].get().strip()

            if value == "":
                messagebox.showwarning("Missing Input",
                                      f"⚠️ Please enter a value for {feature}!")
                return

            min_val, max_val = ranges[feature]
            field_name = feature.capitalize()
            if feature == 'temperature':
                field_name = 'Temperature (°C)'
            elif feature == 'humidity':
                field_name = 'Humidity (%)'
            elif feature == 'rainfall':
                field_name = 'Rainfall (mm)'
            elif feature == 'ph':
                field_name = 'pH'

            valid, num = self.validate_input(value, field_name, min_val, max_val)
            if not valid:
                return

            input_values.append(num)

        try:
            input_array = np.array(input_values).reshape(1, -1)

            prediction_encoded = self.model.predict(input_array)[0]
            predicted_crop = self.label_encoder.inverse_transform([prediction_encoded])[0]

            probabilities = self.model.predict_proba(input_array)[0]
            confidence = probabilities[prediction_encoded] * 100

            self.result_label.configure(text=f"🌾 Recommended Crop: {predicted_crop.upper()}")
            self.confidence_label.configure(text=f"Confidence: {confidence:.2f}%")

            self.prediction_context = {
                'crop': predicted_crop,
                'values': input_values
            }
            self.clear_chat_history()

            if confidence > 80:
                self.result_label.configure(text_color=("green4", "#4ade80"))
            elif confidence > 60:
                self.result_label.configure(text_color=("orange3", "#fb923c"))
            else:
                self.result_label.configure(text_color=("red3", "#f87171"))

            self.update_explanation_text("Generating explanation... Please wait.")
            self.show_loading()
            threading.Thread(
                target=self.generate_explanation,
                args=(predicted_crop, input_values),
                daemon=True
            ).start()

        except Exception as e:
            self.hide_loading()
            self.update_explanation_text("")
            messagebox.showerror("Prediction Error",
                                f"An error occurred during prediction:\n{str(e)}")

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, "end")
        self.result_label.configure(text="🌿 Predicted Crop: --", text_color=("gray40", "gray70"))
        self.confidence_label.configure(text="")
        self.update_explanation_text("Enter soil parameters and click Predict to see the AI explanation.")
        self.prediction_context = None
        self.clear_chat_history()


# ============ MAIN EXECUTION ============
def main():
    print("🌾 Initializing Crop Recommendation System...")

    model, label_encoder, accuracy = train_model()

    if model is None:
        print("Failed to load model. Please check your dataset and model files.")
        return

    print(f"✅ Model loaded successfully! Accuracy: {accuracy*100:.2f}%")
    print(f"📊 Number of crops: {len(label_encoder.classes_)}")

    print("⏳ Loading LLM model...")
    llm = load_llm_model()
    if llm is None:
        print("⚠️ LLM is not available. Explanations will be disabled.")

    print("🚀 Launching GUI...")

    root = ctk.CTk()
    app = CropRecommendationApp(root, model, label_encoder, accuracy, llm=llm)
    root.mainloop()


if __name__ == "__main__":
    main()