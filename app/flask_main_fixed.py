"""
SleepSense Flask API
=====================
Team CC26-PSU230 | Coding Camp 2026 DBS Foundation

Endpoints:
  POST /predict  -> Risk classification dari model TF
  POST /chat     -> Gemini Flash via LangChain
  POST /analyze  -> Predict + Chat dalam satu request
  GET  /health   -> Health check
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─── PATHS ───────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "sleepsense_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_params.json")
META_PATH   = os.path.join(MODELS_DIR, "feature_meta.json")

# ─── CUSTOM TF COMPONENTS ────────────────────────────────────
class AttentionScaling(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attention_dense = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return inputs * self.attention_dense(inputs)

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.units
        return cfg


class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce    = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        at     = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        return tf.reduce_mean(at * tf.pow(1.0 - p_t, self.gamma) * bce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


# ─── SINGLETON LOADER ────────────────────────────────────────
_model         = None
_scaler_params = None
_meta          = None
_llm_chain     = None


def get_model():
    global _model
    if _model is None:
        logger.info("Loading TensorFlow model...")
        _model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "AttentionScaling": AttentionScaling,
                "FocalLoss": FocalLoss
            }
        )
        logger.info("Model loaded.")
    return _model


def get_artifacts():
    global _scaler_params, _meta
    if _scaler_params is None:
        with open(SCALER_PATH) as f:
            _scaler_params = json.load(f)
    if _meta is None:
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _scaler_params, _meta


def get_llm_chain():
    global _llm_chain
    if _llm_chain is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0.7,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Kamu adalah asisten kesehatan SleepSense yang empatik dan suportif. "
             "Tugasmu memberikan saran terkait pola tidur, screen time, dan manajemen stres. "
             "Selalu gunakan Bahasa Indonesia yang hangat. "
             "Output adalah SCREENING AWAL, bukan diagnosis medis. "
             "Berikan 2-3 saran konkret dan tutup dengan kalimat motivasi singkat."),
            ("human",
             "Data pengguna:\n"
             "- Usia: {age} tahun, Jenis kelamin: {gender}\n"
             "- Durasi tidur: {sleep_duration_hours} jam/malam\n"
             "- Kualitas tidur: {sleep_quality_score}/10\n"
             "- Screen time harian: {daily_screen_time_hours} jam\n"
             "- Screen time sebelum tidur: {pre_sleep_screen_time_hours} jam\n"
             "- Aktivitas fisik: {physical_activity_minutes} menit/hari\n"
             "- Kafein: {caffeine_intake_cups} cangkir/hari\n"
             "- Kelelahan mental: {mental_fatigue_score}/10\n"
             "- Notifikasi/hari: {notifications_received_per_day}\n\n"
             "Hasil screening: Risiko {risk_level} (probabilitas: {risk_probability})\n\n"
             "Berikan respons empatik dan saran personal.")
        ])

        _llm_chain = prompt | llm | StrOutputParser()
        logger.info("LangChain + Gemini Flash chain ready.")
    return _llm_chain


# ─── PREPROCESSING ───────────────────────────────────────────
def preprocess(data: dict) -> np.ndarray:
    scaler_params, meta = get_artifacts()
    age       = float(data.get("age", 25))
    age_label = "teen" if age <= 18 else ("young_adult" if age <= 35 else "adult")

    row = {
        "age":                             age,
        "gender":                          float(meta["gender_map"].get(data.get("gender", "Male"), 0)),
        "sleep_duration_hours":            float(data.get("sleep_duration_hours", 7.0)),
        "sleep_quality_score":             float(data.get("sleep_quality_score", 5.0)),
        "daily_screen_time_hours":         float(data.get("daily_screen_time_hours", 4.0)),
        "pre_sleep_screen_time_hours":     float(data.get("pre_sleep_screen_time_hours", 1.0)),
        "physical_activity_minutes":       float(data.get("physical_activity_minutes", 30.0)),
        "caffeine_intake_cups":            float(data.get("caffeine_intake_cups", 2.0)),
        "mental_fatigue_score":            float(data.get("mental_fatigue_score", 5.0)),
        "notifications_received_per_day":  float(data.get("notifications_received_per_day", 50.0)),
        "age_adult":       1.0 if age_label == "adult"       else 0.0,
        "age_teen":        1.0 if age_label == "teen"        else 0.0,
        "age_young_adult": 1.0 if age_label == "young_adult" else 0.0,
    }

    vec   = np.array([row[f] for f in scaler_params["feature_cols"]], dtype=np.float32)
    mean  = np.array(scaler_params["mean"],  dtype=np.float32)
    scale = np.array(scaler_params["scale"], dtype=np.float32)
    return ((vec - mean) / scale).reshape(1, -1)


def interpret(prob: float) -> dict:
    if prob < 0.35:
        level, label = "Rendah", "No Risk"
        summary = "Pola tidur dan gaya hidup Anda tergolong sehat."
    elif prob < 0.65:
        level, label = "Sedang", "Moderate Risk"
        summary = "Ada beberapa aspek gaya hidup yang perlu diperhatikan."
    else:
        level, label = "Tinggi", "At Risk"
        summary = "Pola tidur dan screen time Anda memerlukan perhatian segera."

    return {
        "risk_label":       label,
        "risk_level":       level,
        "risk_probability": round(float(prob), 4),
        "summary":          summary,
        "disclaimer":       "Ini adalah screening awal, BUKAN diagnosis medis."
    }


# ─── ENDPOINTS ───────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "service": "SleepSense Flask API",
        "team":    "CC26-PSU230",
        "version": "1.0.0"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediksi risiko stres dari TensorFlow model.

    Request Body:
    {
        "age": 22, "gender": "Male",
        "sleep_duration_hours": 5.5, "sleep_quality_score": 4.0,
        "daily_screen_time_hours": 8.0, "pre_sleep_screen_time_hours": 2.5,
        "physical_activity_minutes": 15, "caffeine_intake_cups": 4,
        "mental_fatigue_score": 7.5, "notifications_received_per_day": 120
    }

    Response Body:
    {
        "risk_label": "At Risk", "risk_level": "Tinggi",
        "risk_probability": 0.82, "summary": "...", "disclaimer": "..."
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body tidak valid."}), 400

        model  = get_model()
        x      = preprocess(data)
        prob   = float(model(x, training=False).numpy().flatten()[0])
        result = interpret(prob)

        logger.info(f"Predict: prob={prob:.4f} level={result['risk_level']}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Respons empatik dari Gemini Flash via LangChain.

    Request Body: sama seperti /predict + tambahkan risk_level dan risk_probability
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body tidak valid."}), 400

        chain   = get_llm_chain()
        message = chain.invoke({
            "age":                            data.get("age", 25),
            "gender":                         data.get("gender", "Male"),
            "sleep_duration_hours":           data.get("sleep_duration_hours", 7.0),
            "sleep_quality_score":            data.get("sleep_quality_score", 5.0),
            "daily_screen_time_hours":        data.get("daily_screen_time_hours", 4.0),
            "pre_sleep_screen_time_hours":    data.get("pre_sleep_screen_time_hours", 1.0),
            "physical_activity_minutes":      data.get("physical_activity_minutes", 30.0),
            "caffeine_intake_cups":           data.get("caffeine_intake_cups", 2.0),
            "mental_fatigue_score":           data.get("mental_fatigue_score", 5.0),
            "notifications_received_per_day": data.get("notifications_received_per_day", 50.0),
            "risk_level":                     data.get("risk_level", "Sedang"),
            "risk_probability":               data.get("risk_probability", 0.5),
        })

        return jsonify({"message": message}), 200

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Endpoint utama: Predict + Gemini Chat dalam satu request.

    Request Body: sama seperti /predict
    Response Body:
    {
        "prediction": { risk_label, risk_level, risk_probability, summary, disclaimer },
        "advice": "Respons empatik dari Gemini..."
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body tidak valid."}), 400

        # Step 1: TF Model prediction
        model  = get_model()
        x      = preprocess(data)
        prob   = float(model(x, training=False).numpy().flatten()[0])
        result = interpret(prob)

        # Step 2: Gemini advice via LangChain
        chain   = get_llm_chain()
        message = chain.invoke({
            "age":                            data.get("age", 25),
            "gender":                         data.get("gender", "Male"),
            "sleep_duration_hours":           data.get("sleep_duration_hours", 7.0),
            "sleep_quality_score":            data.get("sleep_quality_score", 5.0),
            "daily_screen_time_hours":        data.get("daily_screen_time_hours", 4.0),
            "pre_sleep_screen_time_hours":    data.get("pre_sleep_screen_time_hours", 1.0),
            "physical_activity_minutes":      data.get("physical_activity_minutes", 30.0),
            "caffeine_intake_cups":           data.get("caffeine_intake_cups", 2.0),
            "mental_fatigue_score":           data.get("mental_fatigue_score", 5.0),
            "notifications_received_per_day": data.get("notifications_received_per_day", 50.0),
            "risk_level":                     result["risk_level"],
            "risk_probability":               result["risk_probability"],
        })

        logger.info(f"Analyze: prob={prob:.4f} level={result['risk_level']}")
        return jsonify({
            "prediction": result,
            "advice":     message
        }), 200

    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
