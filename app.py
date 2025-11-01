from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

# ======== SETUP PATH =========
MODEL_PATH = "stroke_model.pkl"
DATA_PATH = "healthcare-dataset-stroke-data.csv"
STATIC_DIR = "static"

# ======== LOAD MODEL =========
model = None
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {e}")

# ======== BUAT GRAFIK OTOMATIS =========
def generate_graphics():
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)
        df = pd.read_csv(DATA_PATH)

        # ===== Grafik Distribusi Stroke =====
        plt.figure(figsize=(5, 4))
        sns.countplot(x="stroke", data=df, palette="coolwarm")
        plt.title("Distribusi Kasus Stroke")
        plt.xlabel("Status Stroke (0=Tidak, 1=Ya)")
        plt.ylabel("Jumlah")
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/distribusi.png")
        plt.close()

        # ===== Grafik Confusion Matrix =====
        if model is not None:
            X = df.drop(columns=["stroke", "id"])
            y = df["stroke"]
            try:
                y_pred = model.predict(X)
                cm = confusion_matrix(y, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix Model Stroke")
                plt.xlabel("Prediksi")
                plt.ylabel("Aktual")
                plt.tight_layout()
                plt.savefig(f"{STATIC_DIR}/confusion.png")
                plt.close()
            except Exception as e:
                print(f"[WARNING] Gagal buat confusion matrix: {e}")

        print("✅ Grafik otomatis berhasil dibuat!")
    except Exception as e:
        print(f"[ERROR] Gagal generate grafik: {e}")

# Panggil fungsi generate grafik saat server mulai
generate_graphics()


# ======== ROUTES =========

@app.route('/')
def home():
    image_paths = []
    if os.path.exists("static/distribusi.png"):
        image_paths.append("/static/distribusi.png")
    if os.path.exists("static/confusion.png"):
        image_paths.append("/static/confusion.png")

    accuracy = 91.2  # contoh nilai akurasi
    return render_template("index.html", image_paths=image_paths, accuracy=accuracy)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="Model belum dimuat!")

    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])

        # Kolom numerik
        numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Kolom dummy 'id' agar sesuai dengan data latih
        df["id"] = 0

        # Prediksi
        prediction = model.predict(df)[0]
        result = "🧠 Berisiko Stroke" if prediction == 1 else "💪 Tidak Berisiko Stroke"

        image_paths = []
        if os.path.exists("static/distribusi.png"):
            image_paths.append("/static/distribusi.png")
        if os.path.exists("static/confusion.png"):
            image_paths.append("/static/confusion.png")

        accuracy = 91.2

        return render_template("index.html", prediction=result, image_paths=image_paths, accuracy=accuracy)

    except Exception as e:
        print(f"[ERROR] Saat prediksi: {e}")
        return render_template('index.html', prediction="Terjadi kesalahan saat prediksi!")


if __name__ == '__main__':
    app.run(debug=True)
