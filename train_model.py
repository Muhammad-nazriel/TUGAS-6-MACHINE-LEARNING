import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

# ===============================
# 1️⃣ LOAD DATA
# ===============================
print("Memuat dataset...")
df = pd.read_csv("train.csv")

# ===============================
# 2️⃣ DATA PREPROCESSING
# ===============================
print("Melakukan preprocessing data...")

# Hapus kolom yang tidak diperlukan
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Isi missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S')  # Isi dengan modus

# Pisahkan fitur dan target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Tentukan kolom kategori dan numerik
categorical_cols = ['Sex', 'Embarked']
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# ===============================
# 3️⃣ PIPELINE: ENCODING + SCALING + MODEL
# ===============================
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numerical_cols)
    ]
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Gabungkan dalam pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ===============================
# 4️⃣ SPLIT DATA & TRAINING
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# ===============================
# 5️⃣ EVALUASI MODEL
# ===============================
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

# ROC & AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Simpan ROC Curve ke file
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Titanic Survival Prediction")
plt.legend()
plt.grid(True)
plt.savefig('static/roc_auc.png')
plt.close()

import os

# Grafik distribusi penumpang berdasarkan kelangsungan hidup
plt.figure(figsize=(5, 4))
sns.countplot(x='Survived', data=df, hue='Survived', palette='Set2', legend=False)
plt.title('Distribusi Penumpang Berdasarkan Kelangsungan Hidup')
plt.savefig('static/distribusi.png')
plt.close()


# Confusion matrix visual
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.savefig('static/confusion.png')
plt.close()


# Feature Importance
rf_model = clf.named_steps["model"]
feature_columns = list(clf.named_steps["preprocessor"].get_feature_names_out())
importances = rf_model.feature_importances_

# Buat DataFrame untuk feature importance
feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# Visualisasi Feature Importance
plt.figure(figsize=(8, 6))
sns.barplot(data=feature_importance_df, y='feature', x='importance', hue='feature', palette='viridis', legend=False)
plt.title("Feature Importance - Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()


# ===============================
# 6️⃣ SIMPAN MODEL DAN OBJEK LAINNYA
# ===============================
# Simpan pipeline model (sudah termasuk encoder dan scaler)
joblib.dump(clf, "titanic_model.pkl")

# Simpan kolom fitur
feature_columns = list(clf.named_steps["preprocessor"].get_feature_names_out())
joblib.dump(feature_columns, "feature_columns.pkl")

# Simpan encoder dan scaler secara terpisah (opsional)
encoder = clf.named_steps["preprocessor"].named_transformers_["cat"]
scaler = clf.named_steps["preprocessor"].named_transformers_["num"]

joblib.dump(encoder, "titanic_encoder.pkl")
joblib.dump(scaler, "titanic_scaler.pkl")

print("\n[OK] Model prediksi kelangsungan hidup Titanic dan semua file pendukung berhasil disimpan!")
print("File yang disimpan:")
print(f"- Model: titanic_model.pkl")
print(f"- Daftar fitur: feature_columns.pkl")
print(f"- Encoder: titanic_encoder.pkl")
print(f"- Scaler: titanic_scaler.pkl")
print("\nSemua grafik berhasil disimpan!")

