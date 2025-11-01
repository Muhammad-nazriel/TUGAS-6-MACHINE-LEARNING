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
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# ===============================
# 2️⃣ DATA PREPROCESSING
# ===============================
# Hapus kolom id
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Isi missing values
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# Pisahkan fitur dan target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Tentukan kolom kategori dan numerik
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

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

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stroke Prediction")
plt.legend()
plt.show()

import os

# Grafik distribusi stroke
plt.figure(figsize=(5,4))
sns.countplot(x='stroke', data=df, palette='Set2')
plt.title('Distribusi Kasus Stroke')
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


# ===============================
# 6️⃣ SIMPAN MODEL DAN OBJEK LAINNYA
# ===============================
# Simpan pipeline model (sudah termasuk encoder dan scaler)
joblib.dump(clf, "model_stroke_rf.pkl")

# Simpan kolom fitur
feature_columns = list(clf.named_steps["preprocessor"].get_feature_names_out())
joblib.dump(feature_columns, "feature_columns.pkl")

# Simpan encoder dan scaler secara terpisah (opsional)
encoder = clf.named_steps["preprocessor"].named_transformers_["cat"]
scaler = clf.named_steps["preprocessor"].named_transformers_["num"]

joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model dan semua file pendukung berhasil disimpan!")
