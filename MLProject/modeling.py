import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import os

dataPath = 'stunting_wasting_preprocessing.csv'
experimentName = 'Stunting Classification - XGBoost'

def run_model():
    # Tracking MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experimentName)

    # 🔹 AKTIFKAN AUTOLOG
    mlflow.sklearn.autolog()

    print('Training dimulai...')
    print('Working directory:', os.getcwd())

    # Load data
    try:
        df = pd.read_csv(dataPath)
        print('Data berhasil diload')
    except FileNotFoundError:
        print(f"Error: Dataset tidak ditemukan di {dataPath}")
        return

    print('Kolom dataset:', df.columns.tolist())

    # Pisahkan fitur dan target
    X = df.drop(columns=['Stunting'])
    y = df['Stunting']

    print('Distribusi kelas:\n', y.value_counts())

    # Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print('Data berhasil di split')

    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('Data berhasil di scale')

    # Training model
    with mlflow.start_run() as run:
        print('Memulai pelatihan model...')

        # Manual log parameter (pelengkap autolog)
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("data_path", dataPath)

        # Model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss"
        )

        model.fit(X_train_scaled, y_train)

        # Evaluasi manual
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Manual log metric (tambahan dari autolog)
        mlflow.log_metric("accuracy_manual", acc)
        mlflow.log_metric("f1_score_weighted", f1)

        # Manual log model (karena autolog log_models=False)
        mlflow.sklearn.log_model(model, "model")

        print(f"Akurasi test set: {acc:.4f}")
        print(f"F1-score (weighted): {f1:.4f}")
        print(f"Pelatihan selesai. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    run_model()
