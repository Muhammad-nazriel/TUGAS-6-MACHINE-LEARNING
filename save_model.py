import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# Load dataset Titanic
print("Memuat dataset Titanic...")
df = pd.read_csv("train.csv")

# Preprocessing data
print("Melakukan preprocessing...")
# Isi missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('S')  # Isi dengan modus

# Pilih kolom yang akan digunakan: Pclass, Age, Sex, Fare, Embarked
X = df[['Pclass', 'Age', 'Sex', 'Fare', 'Embarked']]
y = df['Survived']

# Tentukan kolom kategori dan numerik
categorical_cols = ['Sex', 'Embarked']
numerical_cols = ['Pclass', 'Age', 'Fare']

# Buat preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
])

# Buat pipeline dengan model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

print("Melatih model...")
model.fit(X, y)

# Simpan model
joblib.dump(model, 'titanic_model.pkl')
print("[OK] Model disimpan sebagai titanic_model.pkl")

# Simpan nama kolom fitur untuk referensi
feature_names = list(model.named_steps['preprocessor'].get_feature_names_out())
joblib.dump(feature_names, 'feature_columns.pkl')
print("[OK] Feature columns disimpan sebagai feature_columns.pkl")
