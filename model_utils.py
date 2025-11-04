import pandas as pd
import joblib

# Coba load feature_columns dari model, jika tidak ada akan dibuat otomatis
try:
    feature_columns = joblib.load("feature_columns.pkl")
except:
    # Default feature columns untuk Titanic (setelah OneHotEncoder)
    feature_columns = [
        "num__Pclass", "num__Age", "num__Fare",
        "cat__Sex_male", "cat__Embarked_Q", "cat__Embarked_S"
    ]

# Mapping untuk kolom kategorikal Titanic
label_maps = {
    "Sex": {"female": "female", "male": "male", "Female": "female", "Male": "male"},
    "Embarked": {"S": "S", "C": "C", "Q": "Q"}
}

def preprocess_input(data):
    """
    Preprocess input data untuk prediksi Titanic Survival
    
    Args:
        data: Dictionary dengan keys: Pclass, Age, Sex, Fare, Embarked
    
    Returns:
        DataFrame yang siap untuk diprediksi model
    """
    df = pd.DataFrame([data])
    
    # Pastikan kolom yang diperlukan ada
    required_cols = ['Pclass', 'Age', 'Sex', 'Fare', 'Embarked']
    for col in required_cols:
        if col not in df.columns:
            if col == 'Age':
                df[col] = df.get('Age', 30.0)  # Default age
            elif col == 'Fare':
                df[col] = df.get('Fare', 30.0)  # Default fare
            elif col == 'Embarked':
                df[col] = df.get('Embarked', 'S')  # Default embarked
            elif col == 'Sex':
                df[col] = df.get('Sex', 'male')  # Default sex
            elif col == 'Pclass':
                df[col] = df.get('Pclass', 3)  # Default pclass
    
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median() if 'Age' in df.columns else 30.0)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median() if 'Fare' in df.columns else 30.0)
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Sex'] = df['Sex'].fillna('male')
    df['Pclass'] = df['Pclass'].fillna(3)
    
    # Normalisasi format Sex dan Embarked
    df['Sex'] = df['Sex'].str.lower().map({'female': 'female', 'male': 'male'}).fillna('male')
    df['Embarked'] = df['Embarked'].str.upper().map({'S': 'S', 'C': 'C', 'Q': 'Q'}).fillna('S')
    
    # Pilih hanya kolom yang diperlukan
    df = df[required_cols]
    
    return df


def load_model(model_path="titanic_model.pkl"):
    """
    Load model Random Forest dari file .pkl
    
    Args:
        model_path: Path ke file model (.pkl)
    
    Returns:
        Model yang sudah di-load atau None jika gagal
    """
    try:
        model = joblib.load(model_path)
        print(f"[OK] Model berhasil dimuat dari {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return None


def predict_survival(model, input_data):
    """
    Lakukan prediksi survival Titanic
    
    Args:
        model: Model yang sudah di-load
        input_data: Dictionary dengan data input (Pclass, Age, Sex, Fare, Embarked)
    
    Returns:
        tuple: (prediction, probability)
        - prediction: 0 (Tidak Selamat) atau 1 (Selamat)
        - probability: Probabilitas selamat (0.0 - 1.0)
    """
    if model is None:
        print("[ERROR] Model tidak tersedia")
        return None, 0.0

    try:
        # Preprocess input
        df = preprocess_input(input_data)
        
        # Lakukan prediksi
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return prediction, probability
    except Exception as e:
        print(f"[ERROR] Saat prediksi: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0


# Alias untuk backward compatibility (opsional)
def predict_stroke(model, input_data):
    """Alias untuk predict_survival (backward compatibility)"""
    return predict_survival(model, input_data)
