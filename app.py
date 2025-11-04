from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

# ======== SETUP PATH =========
MODEL_PATH = "titanic_model.pkl"
DATA_PATH = "train.csv"
STATIC_DIR = "static"

# ======== LOAD MODEL =========
model = None
try:
    model = joblib.load(MODEL_PATH)
    print("[OK] Model berhasil dimuat!")
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {e}")

# ======== BUAT GRAFIK OTOMATIS =========
def generate_graphics():
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)
        df = pd.read_csv(DATA_PATH)

        # ===== Grafik Distribusi Kelas =====
        plt.figure(figsize=(5, 4))
        sns.countplot(x="Survived", data=df, hue="Survived", palette="coolwarm", legend=False)
        plt.title("Distribusi Kelas Target (Survived)")
        plt.xlabel("Status (0=Tidak Selamat, 1=Selamat)")
        plt.ylabel("Jumlah")
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/distribusi.png")
        plt.close()

        # ===== Grafik Confusion Matrix =====
        if model is not None:
            # Preprocess data untuk prediksi
            df_clean = df.copy()
            df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
            df_clean['Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())
            df_clean['Embarked'] = df_clean['Embarked'].fillna('S')
            
            # Pilih fitur yang akan digunakan
            features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            X = df_clean[features]
            y = df_clean["Survived"]
            
            # Model pipeline akan preprocess otomatis, tidak perlu get_dummies manual
            try:
                y_pred = model.predict(X)
                cm = confusion_matrix(y, y_pred)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix Model Titanic")
                plt.xlabel("Prediksi")
                plt.ylabel("Aktual")
                plt.tight_layout()
                plt.savefig(f"{STATIC_DIR}/confusion.png")
                plt.close()
            except Exception as e:
                print(f"[WARNING] Gagal buat confusion matrix: {e}")

        print("[OK] Grafik otomatis berhasil dibuat!")
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
    if os.path.exists("static/roc_auc.png"):
        image_paths.append("/static/roc_auc.png")
    if os.path.exists("static/feature_importance.png"):
        image_paths.append("/static/feature_importance.png")

    accuracy = 82.0  # Akurasi dari hasil training
    return render_template("index.html", image_paths=image_paths, accuracy=accuracy)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model tidak tersedia. Silakan latih model terlebih dahulu."
    
    try:
        # Ambil data dari form
        data = {
            'Pclass': int(request.form.get('Pclass')),
            'Sex': request.form.get('Sex'),
            'Age': float(request.form.get('Age', 0)),
            'SibSp': int(request.form.get('SibSp', 0)),
            'Parch': int(request.form.get('Parch', 0)),
            'Fare': float(request.form.get('Fare', 0)),
            'Embarked': request.form.get('Embarked', 'S')
        }
        
        # Konversi ke DataFrame
        # Model pipeline memerlukan kolom asli sebelum preprocessing
        input_df = pd.DataFrame([data])
        
        # Pastikan kolom yang diperlukan ada (sesuai dengan yang digunakan saat training)
        required_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']
        for col in required_cols:
            if col not in input_df.columns:
                if col == 'Age':
                    input_df[col] = 30.0
                elif col == 'Fare':
                    input_df[col] = 30.0
                elif col == 'Embarked':
                    input_df[col] = 'S'
                elif col == 'Sex':
                    input_df[col] = 'male'
                elif col == 'Pclass':
                    input_df[col] = 3
                else:
                    input_df[col] = 0
        
        # Pilih hanya kolom yang diperlukan
        input_df = input_df[required_cols]
        
        # Lakukan prediksi (model pipeline akan preprocess otomatis)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100
        
        # Tentukan pesan hasil
        result_text = "Selamat! Anda diprediksi akan selamat." if prediction == 1 else "Maaf, Anda diprediksi tidak selamat."
        
        # Siapkan path gambar
        image_paths = []
        if os.path.exists("static/distribusi.png"):
            image_paths.append("/static/distribusi.png")
        if os.path.exists("static/confusion.png"):
            image_paths.append("/static/confusion.png")
        if os.path.exists("static/roc_auc.png"):
            image_paths.append("/static/roc_auc.png")
        if os.path.exists("static/feature_importance.png"):
            image_paths.append("/static/feature_importance.png")

        accuracy = 82.0  # Akurasi dari hasil training
        
        return render_template("index.html", 
                             prediction=result_text,
                             probability=f"{probability:.2f}",
                             image_paths=image_paths,
                             accuracy=accuracy,
                             input_data=data)
    except Exception as e:
        print(f"[ERROR] Saat prediksi: {e}")
        return render_template('index.html', prediction=f"Terjadi kesalahan: {str(e)}")


import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

