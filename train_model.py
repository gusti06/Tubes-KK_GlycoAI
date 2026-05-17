"""
Script untuk training machine learning model menggunakan dataset diabetes.

Model yang dihasilkan digunakan oleh aplikasi GlycoAI untuk prediksi alternatif
dibanding Fuzzy Logic. Support untuk SVM dan Random Forest.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from statistics import median
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"
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


def load_and_clean_data() -> tuple[pd.DataFrame, List[Dict[str, float]]]:
    """Load CSV dan clean missing values dengan median imputation."""
    rows: List[Dict[str, str]] = []
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")

    with DATASET_PATH.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append({str(key).strip().lower(): value for key, value in row.items()})

    if not rows:
        raise ValueError("Dataset kosong")

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
    for row in rows:
        cleaned_row: Dict[str, float] = {}
        for field in NUMERIC_FIELDS:
            value = row.get(field)
            if _is_missing_value(field, value):
                cleaned_row[field] = float(medians[field])
            else:
                try:
                    cleaned_row[field] = float(value)
                except (TypeError, ValueError):
                    cleaned_row[field] = float(medians[field])
        cleaned_rows.append(cleaned_row)

    df = pd.DataFrame(cleaned_rows)
    return df, cleaned_rows


def train_model(df: pd.DataFrame, model_type: str = "random_forest") -> Dict[str, object]:
    """Train machine learning model dan return metrics."""
    
    X = df.drop("outcome", axis=1)
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "svm":
        model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    report_data = {
        "model_type": model_type,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(report.get("1", report.get(1, {})).get("f1-score", 0)), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": {k: v for k, v in report.items() if k in ["0", "1", 0, 1, "accuracy", "macro avg", "weighted avg"]},
        "training_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": len(df),
        "positive_samples": int(y.sum()),
        "negative_samples": int((y == 0).sum()),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    with open(TRAINING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data


def main():
    """Jalankan training dan simpan model."""
    print("📊 Loading dan cleaning dataset...")
    df, _ = load_and_clean_data()
    print(f"✓ Dataset loaded: {len(df)} samples")

    print("\n🤖 Training Random Forest model...")
    report = train_model(df, model_type="random_forest")

    print("\n✅ Training complete!")
    print(f"   Accuracy:  {report['accuracy']}")
    print(f"   Precision: {report['precision']}")
    print(f"   Recall:    {report['recall']}")
    print(f"   F1-Score:  {report['f1_score']}")
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print(f"📋 Report saved to: {TRAINING_REPORT_PATH}")


if __name__ == "__main__":
    main()
