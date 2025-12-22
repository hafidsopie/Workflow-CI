import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_model():
    mlflow.set_experiment("Stunting Classification - XGBoost")

    with mlflow.start_run() as run:
        print("Training dimulai...")

        data = pd.read_csv("stunting_wasting_preprocessed.csv")
        X = data.drop("status_gizi", axis=1)
        y = data["status_gizi"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBClassifier(eval_metric="logloss")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.xgboost.log_model(model, artifact_path="model")

        # 🔥 SIMPAN run_id
        print(f"RUN_ID={run.info.run_id}")

if __name__ == "__main__":
    run_model()
