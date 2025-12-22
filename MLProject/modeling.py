import os
import pandas as pd
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


DATA_PATH = "stunting_wasting_preprocessing.csv"
EXPERIMENT_NAME = "Stunting Classification - XGBoost"
TARGET_COL = "Stunting"
RANDOM_STATE = 42


def run_model():
    print("Training dimulai...")
    print("Working directory:", os.getcwd())


    df = pd.read_csv(DATA_PATH)
    print("Data berhasil diload")
    print("Kolom dataset:", df.columns.tolist())
    print("Distribusi kelas:\n", df[TARGET_COL].value_counts())

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print("Data berhasil di split")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Data berhasil di scale")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score_weighted", f1)

    print(f"Akurasi  : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.xgboost.autolog()

    run_model()
