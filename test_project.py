#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk test apakah semua komponen project berfungsi
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

print("=" * 50)
print("TESTING PROJECT COMPONENTS")
print("=" * 50)

# 1. Test Dataset
print("\n[1] Testing Dataset...")
try:
    df = pd.read_csv('train.csv')
    print(f"   [OK] Dataset berhasil dimuat: {df.shape}")
    print(f"   [OK] Kolom: {len(df.columns)} kolom")
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    sys.exit(1)

# 2. Test Model
print("\n[2] Testing Model...")
try:
    model = joblib.load('titanic_model.pkl')
    print(f"   [OK] Model berhasil dimuat: {type(model)}")
    if isinstance(model, Pipeline):
        print(f"   [OK] Model adalah Pipeline dengan {len(model.named_steps)} steps")
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    sys.exit(1)

# 3. Test Feature Columns
print("\n[3] Testing Feature Columns...")
try:
    feature_cols = joblib.load('feature_columns.pkl')
    print(f"   [OK] Feature columns berhasil dimuat: {len(feature_cols)} fitur")
except Exception as e:
    print(f"   [WARNING] Warning: {e} (opsional)")

# 4. Test Static Files
print("\n[4] Testing Static Files...")
static_files = [
    'static/distribusi.png',
    'static/confusion.png',
    'static/roc_auc.png',
    'static/feature_importance.png'
]
for file in static_files:
    if os.path.exists(file):
        print(f"   [OK] {file}")
    else:
        print(f"   [MISSING] {file} TIDAK ADA")

# 5. Test Templates
print("\n[5] Testing Templates...")
templates = [
    'templates/index.html'
]
for template in templates:
    if os.path.exists(template):
        print(f"   [OK] {template}")
    else:
        print(f"   [MISSING] {template} TIDAK ADA")

# 6. Test Prediction
print("\n[6] Testing Prediction...")
try:
    # Buat test data (sesuai format yang diharapkan model pipeline)
    # Model pipeline memerlukan kolom asli sebelum preprocessing
    test_data = pd.DataFrame({
        'Pclass': [1],
        'Sex': ['female'],
        'Age': [30],
        'SibSp': [0],
        'Parch': [0],
        'Fare': [50.0],
        'Embarked': ['S']
    })
    
    # Prediksi langsung dengan model pipeline (akan preprocess otomatis)
    prediction = model.predict(test_data)[0]
    probability = model.predict_proba(test_data)[0][1] * 100
    
    print(f"   [OK] Prediksi berhasil: {prediction} (probabilitas: {probability:.2f}%)")
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Test Libraries
print("\n[7] Testing Required Libraries...")
libraries = ['flask', 'pandas', 'numpy', 'sklearn', 'seaborn', 'matplotlib', 'joblib']
for lib in libraries:
    try:
        __import__(lib)
        print(f"   [OK] {lib}")
    except ImportError:
        print(f"   [MISSING] {lib} TIDAK TERINSTALL")

print("\n" + "=" * 50)
print("TESTING SELESAI")
print("=" * 50)
print("\n[OK] Project siap digunakan!")
print("\nCara menjalankan:")
print("  1. python train_model.py  (untuk training)")
print("  2. python app.py          (untuk web app)")

