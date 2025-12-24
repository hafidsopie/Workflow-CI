import os
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

# ===============================
# Autolog (BOLEH)
# ===============================
mlflow.sklearn.autolog()

# ===============================
# Training Function
# ===============================
def run_model(data_path):
    print("Memulai training model...")
    print("Mencari dataset di:", data_path)

    df = pd.read_csv(data_path)
    print("Dataset berhasil diload")

    X = df.drop("Stunting", axis=1)
    y = df["Stunting"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    print("Training model XGBoost...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    mlflow.log_metric("accuracy", accuracy)

    print("Training selesai")

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    run_model(args.data_path)
