import pandas as pd
import os
import dagshub
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier


DATA_PATH = "stunting_wasting_preprocessing.csv"
EXPERIMENT_NAME = "Stunting Classification - XGBoost"

DAGSHUB_USER = "hafidsopie"
DAGSHUB_REPO = "Membangun_Model"


def run_tuning():
    dagshub.init(
        repo_owner=DAGSHUB_USER,
        repo_name=DAGSHUB_REPO,
        mlflow=True
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Tuning dimulai...")
    print("Working directory:", os.getcwd())

    # LOAD DATA

    try:
        df = pd.read_csv(DATA_PATH)
        print("Dataset berhasil diload")
    except FileNotFoundError:
        print(f"Dataset tidak ditemukan: {DATA_PATH}")
        return

    X = df.drop(columns=["Stunting"])
    y = df["Stunting"]

    print("Distribusi kelas:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    with mlflow.start_run():

        model = XGBClassifier(
            random_state=42,
            eval_metric="mlogloss"
        )

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_

        print("Best Parameters:", grid.best_params_)


        # EVALUASI
        
        y_pred = best_model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Akurasi Test: {acc:.4f}")
        print(f"F1-score Weighted: {f1:.4f}")


        mlflow.log_param("model", "XGBoost_Tuned")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1_weighted")
        mlflow.log_param("data_path", DATA_PATH)

        for param, value in grid.best_params_.items():
            mlflow.log_param(param, value)

        mlflow.log_metric("accuracy_test", acc)
        mlflow.log_metric("f1_score_weighted_test", f1)

        mlflow.sklearn.log_model(best_model, "best_model")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        # Classification Report
        report = classification_report(y_test, y_pred)
        report_path = "classification_report.txt"

        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_path)

        print("Artefak berhasil dikirim ke DagsHub")
        print("Run selesai")

# ENTRY POINT
if __name__ == "__main__":
    run_tuning()
