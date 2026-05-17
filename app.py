from __future__ import annotations

import csv
import json
import os
import pickle
from statistics import median
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, render_template, request

from fuzzy_logic import predict_diabetes_risk

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"
HISTORY_PATH = BASE_DIR / "models" / "prediction_history.json"
MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "trained_scaler.pkl"
TRAINING_REPORT_PATH = BASE_DIR / "models" / "training_report.json"

ZERO_AS_MISSING_FIELDS = {
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
}
NUMERIC_FIELDS = {
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
    "outcome",
}

app = Flask(__name__)

# Global variables untuk ML model
ML_MODEL = None
ML_SCALER = None
ML_AVAILABLE = False


def load_ml_model():
    """Load trained ML model dan scaler jika tersedia."""
    global ML_MODEL, ML_SCALER, ML_AVAILABLE
    try:
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                ML_MODEL = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                ML_SCALER = pickle.load(f)
            ML_AVAILABLE = True
            print("✓ ML Model loaded successfully")
        else:
            ML_AVAILABLE = False
            print("⚠ ML Model files not found")
    except Exception as e:
        ML_AVAILABLE = False
        print(f"✗ Error loading ML model: {e}")


def ensure_storage() -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("[]", encoding="utf-8")


def load_history() -> List[Dict[str, object]]:
    ensure_storage()
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_history(entries: List[Dict[str, object]]) -> None:
    HISTORY_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def append_history(entry: Dict[str, object]) -> None:
    history = load_history()
    history.append(entry)
    save_history(history)


def load_dataset_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not DATASET_PATH.exists():
        return rows

    with DATASET_PATH.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append({str(key).strip().lower(): value for key, value in row.items()})
    return rows


def _is_missing_value(field_name: str, raw_value: str | None) -> bool:
    if raw_value is None:
        return True

    text_value = str(raw_value).strip()
    if text_value == "" or text_value.lower() in {"na", "n/a", "nan", "null", "none", "?"}:
        return True

    if field_name in ZERO_AS_MISSING_FIELDS:
        try:
            return float(text_value) == 0.0
        except ValueError:
            return True

    return False


def clean_dataset_rows(rows: List[Dict[str, str]]) -> Dict[str, object]:
    if not rows:
        return {"rows": [], "raw_rows": 0, "cleaned_rows": 0, "missing_imputed": 0}

    cleanable_values: Dict[str, List[float]] = {field: [] for field in NUMERIC_FIELDS}
    missing_counts = {field: 0 for field in NUMERIC_FIELDS}

    for row in rows:
        for field in NUMERIC_FIELDS:
            value = row.get(field)
            if _is_missing_value(field, value):
                missing_counts[field] += 1
                continue
            try:
                cleanable_values[field].append(float(value))
            except (TypeError, ValueError):
                missing_counts[field] += 1

    medians = {
        field: median(values) if values else 0.0
        for field, values in cleanable_values.items()
    }

    cleaned_rows: List[Dict[str, float]] = []
    missing_imputed = 0

    for row in rows:
        cleaned_row: Dict[str, float] = {}
        for field in NUMERIC_FIELDS:
            value = row.get(field)
            if _is_missing_value(field, value):
                cleaned_row[field] = float(medians[field])
                missing_imputed += 1
            else:
                try:
                    cleaned_row[field] = float(value)
                except (TypeError, ValueError):
                    cleaned_row[field] = float(medians[field])
                    missing_imputed += 1
        cleaned_rows.append(cleaned_row)

    return {
        "rows": cleaned_rows,
        "raw_rows": len(rows),
        "cleaned_rows": len(cleaned_rows),
        "missing_imputed": missing_imputed,
        "missing_counts": missing_counts,
        "median_values": medians,
    }


def summarize_history() -> Dict[str, object]:
    history = load_history()
    counts = {"low": 0, "medium": 0, "high": 0}
    for item in history:
        intensity = str(item.get("intensity") or item.get("result", {}).get("intensity", "low"))
        if intensity in counts:
            counts[intensity] += 1

    total = len(history)
    return {
        "total": total,
        "low": counts["low"],
        "medium": counts["medium"],
        "high": counts["high"],
        "low_percent": round((counts["low"] / total) * 100, 1) if total else 0,
        "medium_percent": round((counts["medium"] / total) * 100, 1) if total else 0,
        "high_percent": round((counts["high"] / total) * 100, 1) if total else 0,
    }


def dataset_overview() -> Dict[str, object]:
    rows = load_dataset_rows()
    cleaned = clean_dataset_rows(rows)
    cleaned_rows = cleaned["rows"]
    if not cleaned_rows:
        return {
            "total_rows": 0,
            "diabetes_percent": 0,
            "avg_glucose": 0,
            "avg_bmi": 0,
            "raw_rows": 0,
            "cleaned_rows": 0,
            "missing_imputed": 0,
            "cleaning_note": "Dataset belum tersedia.",
        }

    total_rows = len(cleaned_rows)
    diabetes_count = 0
    glucose_total = 0.0
    bmi_total = 0.0

    for row in cleaned_rows:
        diabetes_count += 1 if int(row.get("outcome", 0)) == 1 else 0
        glucose_total += float(row.get("glucose", 0) or 0)
        bmi_total += float(row.get("bmi", 0) or 0)

    return {
        "total_rows": total_rows,
        "diabetes_percent": round((diabetes_count / total_rows) * 100, 1),
        "avg_glucose": round(glucose_total / total_rows, 1),
        "avg_bmi": round(bmi_total / total_rows, 1),
        "raw_rows": cleaned["raw_rows"],
        "cleaned_rows": cleaned["cleaned_rows"],
        "missing_imputed": cleaned["missing_imputed"],
        "cleaning_note": "Missing value numerik pada dataset Pima diimputasi dengan median.",
    }


def build_result_payload(form_data: Dict[str, float]) -> Dict[str, object]:
    fuzzy_result = predict_diabetes_risk(form_data)
    return {
        "risk_score": fuzzy_result.risk_score,
        "category": fuzzy_result.category,
        "intensity": fuzzy_result.intensity,
        "explanation": fuzzy_result.explanation,
        "recommendations": fuzzy_result.recommendations,
        "rule_details": fuzzy_result.rule_details,
        "memberships": fuzzy_result.memberships,
    }


def predict_diabetes_ml(form_data: Dict[str, float]) -> Dict[str, object]:
    """Prediksi menggunakan trained ML model RandomForest."""
    if not ML_AVAILABLE or ML_MODEL is None or ML_SCALER is None:
        return {
            "error": "ML Model belum tersedia. Silakan jalankan train_model.py",
            "success": False,
        }

    try:
        import numpy as np
        
        features = np.array([[
            form_data.get("pregnancies", 0),
            form_data.get("glucose", 0),
            form_data.get("blood_pressure", 0),
            form_data.get("skin_thickness", 0),
            form_data.get("insulin", 0),
            form_data.get("bmi", 0),
            form_data.get("diabetes_pedigree_function", 0),
            form_data.get("age", 0),
        ]])
        
        features_scaled = ML_SCALER.transform(features)
        prediction = ML_MODEL.predict(features_scaled)[0]
        probability = ML_MODEL.predict_proba(features_scaled)[0]
        
        risk_percentage = round(float(probability[1]) * 100, 2)
        
        return {
            "success": True,
            "method": "machine_learning",
            "prediction": int(prediction),
            "risk_percentage": risk_percentage,
            "probabilities": {
                "no_diabetes": round(float(probability[0]), 4),
                "diabetes": round(float(probability[1]), 4),
            },
            "category": "Risiko Tinggi" if risk_percentage >= 60 else "Risiko Sedang" if risk_percentage >= 40 else "Risiko Rendah",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@app.route("/")
def index():
    stats = summarize_history()
    dataset_stats = dataset_overview()
    return render_template("index.html", stats=stats, dataset_stats=dataset_stats)


@app.route("/prediksi")
def prediksi():
    stats = summarize_history()
    dataset_stats = dataset_overview()
    return render_template("prediksi.html", stats=stats, dataset_stats=dataset_stats, ml_available=ML_AVAILABLE)


@app.route("/tentang-diabetes")
def tentang_diabetes():
    return render_template("tentang_diabetes.html")


@app.route("/tentang-aplikasi")
def tentang_aplikasi():
    return render_template("tentang_aplikasi.html")


@app.route("/training-results")
def training_results():
    """Halaman untuk menampilkan hasil training ML model."""
    if not TRAINING_REPORT_PATH.exists():
        return render_template("training_results.html", report=None, ml_available=False)
    
    try:
        with open(TRAINING_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        return render_template("training_results.html", report=report, ml_available=True)
    except Exception:
        return render_template("training_results.html", report=None, ml_available=False)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or request.form.to_dict()

    try:
        form_data = {
            "age": float(payload.get("age", 0)),
            "bmi": float(payload.get("bmi", 0)),
            "glucose": float(payload.get("glucose", 0)),
            "blood_pressure": float(payload.get("blood_pressure", 0)),
            "insulin": float(payload.get("insulin", 0)),
            "family_history": float(payload.get("family_history", 0)),
            "activity": float(payload.get("activity", 0)),
        }
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Input tidak valid."}), 400

    result = build_result_payload(form_data)
    history_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": form_data,
        "category": result["category"],
        "intensity": result["intensity"],
        "risk_score": result["risk_score"],
        "result": result,
    }
    append_history(history_entry)

    stats = summarize_history()
    return jsonify(
        {
            "success": True,
            "message": "Prediksi berhasil dihitung.",
            "result": result,
            "stats": stats,
        }
    )


@app.route("/api/predict-ml", methods=["POST"])
def api_predict_ml():
    """API endpoint untuk prediksi menggunakan ML model."""
    if not ML_AVAILABLE:
        return jsonify({
            "success": False,
            "message": "ML Model belum tersedia. Silakan jalankan train_model.py",
        }), 503

    payload = request.get_json(silent=True) or request.form.to_dict()

    try:
        form_data = {
            "pregnancies": float(payload.get("pregnancies", 0)),
            "age": float(payload.get("age", 0)),
            "bmi": float(payload.get("bmi", 0)),
            "glucose": float(payload.get("glucose", 0)),
            "blood_pressure": float(payload.get("blood_pressure", 0)),
            "insulin": float(payload.get("insulin", 0)),
            "skin_thickness": float(payload.get("skin_thickness", 0)),
            "diabetes_pedigree_function": float(payload.get("diabetes_pedigree_function", 0)),
        }
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Input tidak valid."}), 400

    result = predict_diabetes_ml(form_data)
    return jsonify({
        "success": True,
        "message": "Prediksi ML berhasil dihitung.",
        "result": result,
    })


@app.route("/api/stats")
def api_stats():
    stats = summarize_history()
    dataset_stats = dataset_overview()
    return jsonify({"success": True, "stats": stats, "dataset": dataset_stats, "ml_available": ML_AVAILABLE})


@app.route("/api/training-report")
def api_training_report():
    """API endpoint untuk mendapatkan training report."""
    if not TRAINING_REPORT_PATH.exists():
        return jsonify({
            "success": False,
            "message": "Training report tidak ditemukan",
        }), 404

    try:
        with open(TRAINING_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        return jsonify({"success": True, "report": report})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.context_processor
def inject_globals():
    return {
        "app_name": "GlycoAI",
        "year": datetime.now().year,
    }


if __name__ == "__main__":
    ensure_storage()
    load_ml_model()
    app.run(debug=True)
