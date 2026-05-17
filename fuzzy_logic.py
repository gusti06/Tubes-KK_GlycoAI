from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FuzzyResult:
    risk_score: float
    category: str
    intensity: str
    explanation: str
    recommendations: List[str]
    rule_details: List[Dict[str, float]]
    memberships: Dict[str, Dict[str, float]]


def triangular(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def age_membership(age: float) -> Dict[str, float]:
    return {
        "rendah": trapezoid(age, 0, 0, 29, 40),
        "sedang": triangular(age, 30, 45, 60),
        "tinggi": trapezoid(age, 55, 65, 120, 120),
    }


def bmi_membership(bmi: float) -> Dict[str, float]:
    return {
        "rendah": trapezoid(bmi, 0, 0, 19, 23),
        "sedang": triangular(bmi, 21, 27, 33),
        "tinggi": trapezoid(bmi, 30, 35, 60, 60),
    }


def glucose_membership(glucose: float) -> Dict[str, float]:
    return {
        "rendah": trapezoid(glucose, 0, 0, 90, 110),
        "sedang": triangular(glucose, 100, 130, 160),
        "tinggi": trapezoid(glucose, 150, 180, 350, 350),
    }


def blood_pressure_membership(bp: float) -> Dict[str, float]:
    return {
        "rendah": trapezoid(bp, 0, 0, 68, 78),
        "sedang": triangular(bp, 72, 82, 92),
        "tinggi": trapezoid(bp, 88, 95, 140, 140),
    }


def insulin_membership(insulin: float) -> Dict[str, float]:
    return {
        "rendah": trapezoid(insulin, 0, 0, 55, 80),
        "sedang": triangular(insulin, 65, 120, 170),
        "tinggi": trapezoid(insulin, 150, 190, 900, 900),
    }


def family_history_membership(value: float) -> Dict[str, float]:
    value = max(0.0, min(1.0, value))
    return {
        "rendah": trapezoid(value, 0.0, 0.0, 0.2, 0.4),
        "sedang": triangular(value, 0.25, 0.5, 0.75),
        "tinggi": trapezoid(value, 0.6, 0.8, 1.0, 1.0),
    }


def activity_membership(value: float) -> Dict[str, float]:
    value = max(0.0, min(1.0, value))
    return {
        "rendah": trapezoid(value, 0.0, 0.0, 0.25, 0.45),
        "sedang": triangular(value, 0.3, 0.55, 0.8),
        "tinggi": trapezoid(value, 0.65, 0.8, 1.0, 1.0),
    }


def output_low(score: float) -> float:
    return trapezoid(score, 0, 0, 20, 40)


def output_medium(score: float) -> float:
    return triangular(score, 25, 50, 75)


def output_high(score: float) -> float:
    return trapezoid(score, 60, 80, 100, 100)


def infer_rules(inputs: Dict[str, float]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    age = age_membership(inputs["age"])
    bmi = bmi_membership(inputs["bmi"])
    glucose = glucose_membership(inputs["glucose"])
    blood_pressure = blood_pressure_membership(inputs["blood_pressure"])
    insulin = insulin_membership(inputs["insulin"])
    family_history = family_history_membership(inputs["family_history"])
    activity = activity_membership(inputs["activity"])

    memberships = {
        "age": age,
        "bmi": bmi,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "insulin": insulin,
        "family_history": family_history,
        "activity": activity,
    }

    rules = [
        {
            "name": "Glucose tinggi dan BMI tinggi -> risiko tinggi",
            "category": "high",
            "strength": min(glucose["tinggi"], bmi["tinggi"]),
        },
        {
            "name": "Glucose tinggi dan insulin tinggi -> risiko tinggi",
            "category": "high",
            "strength": min(glucose["tinggi"], insulin["tinggi"]),
        },
        {
            "name": "Riwayat keluarga tinggi dan umur tinggi -> risiko tinggi",
            "category": "high",
            "strength": min(family_history["tinggi"], age["tinggi"]),
        },
        {
            "name": "Glucose sedang dan aktivitas rendah -> risiko sedang",
            "category": "medium",
            "strength": min(glucose["sedang"], activity["rendah"]),
        },
        {
            "name": "Tekanan darah tinggi dan umur sedang -> risiko sedang",
            "category": "medium",
            "strength": min(blood_pressure["tinggi"], age["sedang"]),
        },
        {
            "name": "Glucose rendah dan BMI sedang -> risiko rendah",
            "category": "low",
            "strength": min(glucose["rendah"], bmi["sedang"]),
        },
        {
            "name": "Glucose rendah dan aktivitas tinggi -> risiko rendah",
            "category": "low",
            "strength": min(glucose["rendah"], activity["tinggi"]),
        },
        {
            "name": "BMI rendah dan riwayat keluarga rendah -> risiko rendah",
            "category": "low",
            "strength": min(bmi["rendah"], family_history["rendah"]),
        },
        {
            "name": "Banyak indikator sedang -> risiko sedang",
            "category": "medium",
            "strength": min(glucose["sedang"], bmi["sedang"], blood_pressure["sedang"]),
        },
        {
            "name": "Umur tinggi dan BMI tinggi -> risiko tinggi",
            "category": "high",
            "strength": min(age["tinggi"], bmi["tinggi"]),
        },
    ]

    return memberships, rules


def defuzzify(rules: List[Dict[str, float]]) -> float:
    aggregated_low = max((rule["strength"] for rule in rules if rule["category"] == "low"), default=0.0)
    aggregated_medium = max((rule["strength"] for rule in rules if rule["category"] == "medium"), default=0.0)
    aggregated_high = max((rule["strength"] for rule in rules if rule["category"] == "high"), default=0.0)

    numerator = 0.0
    denominator = 0.0

    for score in range(0, 101):
        mu_low = min(aggregated_low, output_low(score))
        mu_medium = min(aggregated_medium, output_medium(score))
        mu_high = min(aggregated_high, output_high(score))
        mu = max(mu_low, mu_medium, mu_high)
        numerator += score * mu
        denominator += mu

    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 2)


def categorize(score: float) -> Tuple[str, str]:
    if score < 40:
        return "Risiko Rendah", "low"
    if score < 70:
        return "Risiko Sedang", "medium"
    return "Risiko Tinggi", "high"


def explain(score: float, category: str, inputs: Dict[str, float]) -> Tuple[str, List[str]]:
    if category == "Risiko Tinggi":
        explanation = (
            "Kombinasi kadar glukosa, BMI, dan faktor risiko pendukung menunjukkan kecenderungan kuat ke kategori tinggi."
        )
        recommendations = [
            "Segera konsultasi ke dokter untuk evaluasi medis lebih lanjut.",
            "Kurangi konsumsi gula sederhana dan makanan tinggi kalori.",
            "Lakukan aktivitas fisik teratur minimal 150 menit per minggu.",
        ]
    elif category == "Risiko Sedang":
        explanation = (
            "Beberapa indikator berada di zona waspada sehingga risiko diabetes tergolong sedang.")
        recommendations = [
            "Pantau pola makan dan berat badan secara konsisten.",
            "Perbanyak aktivitas fisik ringan hingga sedang.",
            "Lakukan pemeriksaan gula darah berkala.",
        ]
    else:
        explanation = (
            "Mayoritas indikator masih berada pada zona aman sehingga risiko diabetes tergolong rendah."
        )
        recommendations = [
            "Pertahankan pola makan seimbang dan aktif bergerak.",
            "Tetap cek kesehatan secara berkala.",
            "Jaga kualitas tidur dan hindari kebiasaan sedentary terlalu lama.",
        ]

    if inputs["family_history"] >= 0.7:
        recommendations.append("Karena ada riwayat keluarga, lakukan skrining lebih rutin.")

    if inputs["activity"] <= 0.35:
        recommendations.append("Tingkatkan aktivitas fisik harian secara bertahap.")

    return explanation, recommendations


def predict_diabetes_risk(inputs: Dict[str, float]) -> FuzzyResult:
    memberships, rules = infer_rules(inputs)
    score = defuzzify(rules)
    category, intensity = categorize(score)
    explanation, recommendations = explain(score, category, inputs)

    rule_details = [
        {
            "name": rule["name"],
            "category": rule["category"],
            "strength": round(rule["strength"], 3),
        }
        for rule in sorted(rules, key=lambda item: item["strength"], reverse=True)
    ]

    return FuzzyResult(
        risk_score=score,
        category=category,
        intensity=intensity,
        explanation=explanation,
        recommendations=recommendations,
        rule_details=rule_details,
        memberships=memberships,
    )
