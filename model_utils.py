import pandas as pd
import joblib

# Coba load feature_columns dari model, jika tidak ada akan dibuat otomatis
try:
    feature_columns = joblib.load("feature_columns.pkl")
except:
    feature_columns = [
        "gender", "age", "hypertension", "heart_disease",
        "ever_married", "work_type", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status"
    ]

# Hapus kolom id jika ada
if "id" in feature_columns:
    feature_columns.remove("id")

# Mapping untuk kolom kategorikal
label_maps = {
    "gender": {"Female": 0, "Male": 1, "Other": 2},
    "ever_married": {"No": 0, "Yes": 1},
    "work_type": {
        "Govt_job": 0,
        "Never_worked": 1,
        "Private": 2,
        "Self-employed": 3,
        "children": 4
    },
    "Residence_type": {"Rural": 0, "Urban": 1},
    "smoking_status": {
        "Unknown": 0,
        "formerly smoked": 1,
        "never smoked": 2,
        "smokes": 3
    }
}

def preprocess_input(data):
    df = pd.DataFrame([data])

    # Mapping kategori ke angka
    for col, mapping in label_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Pastikan semua numerik
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Reindex sesuai kolom model
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# ✅ Tambahkan dua fungsi baru biar app.py bisa jalan
def load_model(model_path="stroke_model.pkl"):
    """Load model Random Forest dari file .pkl"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return None


def predict_stroke(model, input_data):
    """Lakukan prediksi risiko stroke"""
    if model is None:
        return None, 0.0

    try:
        df = preprocess_input(input_data)
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return prediction, probability
    except Exception as e:
        print(f"[ERROR] Saat prediksi: {e}")
        return None, 0.0
