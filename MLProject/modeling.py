import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

# ===============================
# Konfigurasi MLflow
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_PATH = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
mlflow.set_experiment("Prediksi_Balita_Stunting_Wasting_XGBoost")

mlflow.sklearn.autolog()

# ===============================
# Path Dataset
# ===============================
DATA_PATH = os.path.join(BASE_DIR, "stunting_wasting_preprocessing.csv")

# ===============================
# Training Function
# ===============================
def run_model():
    print("Memulai training model...")
    print("Mencari dataset di:", DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    print("Dataset berhasil diload")

    # Fitur dan target
    X = df.drop("Stunting", axis=1)
    y = df["Stunting"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model XGBoost
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",  
        eval_metric="mlogloss",
        random_state=42
    )

    # Training
    print("Training model XGBoost...")
    model.fit(X_train_scaled, y_train)

    # Evaluasi
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    mlflow.log_metric("accuracy", accuracy)

    # Simpan model & scaler ke MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.iloc[:5]
    )

    mlflow.sklearn.log_model(
        sk_model=scaler,
        artifact_path="scaler"
    )

    print("Training selesai dan model berhasil disimpan di MLflow.")

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    os.makedirs(MLRUNS_PATH, exist_ok=True)
    run_model()


