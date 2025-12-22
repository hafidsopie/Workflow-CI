import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import os

DATA_PATH = "stunting_wasting_preprocessing.csv"

def run_model():
    print("Training dimulai...")
    print("Working directory:", os.getcwd())

    # AUTLOG SAJA (JANGAN SET EXPERIMENT, JANGAN START RUN)
    mlflow.sklearn.autolog()

    df = pd.read_csv(DATA_PATH)
    print("Data berhasil diload")

    X = df.drop(columns=["Stunting"])
    y = df["Stunting"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Akurasi: {acc}")
    print(f"F1-score: {f1}")

if __name__ == "__main__":
    run_model()
