import argparse
import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_model(data_path):
    mlflow.set_experiment("Stunting Classification - XGBoost")

    print("Training dimulai...")

    data = pd.read_csv(data_path)
    X = data.drop(columns=["status_gizi"])
    y = data["status_gizi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.xgboost.log_model(model, artifact_path="model")

    print("Training selesai")
    print("Accuracy:", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    run_model(args.data_path)
