import os
import pandas as pd
import joblib
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

FEATURES = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']

def model_train(input_path: str, model_path: str, n_estimators: int = 20):
    print(f"Training with input={input_path}")
    input_file = os.path.join(input_path, "processed.csv") 

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Training data not found: {input_file}")

    df = pd.read_csv(input_file)
    
    if not set(FEATURES).issubset(df.columns) or 'target' not in df.columns:
        raise ValueError("Input must contain iris feature columns and 'target'")

    X = df[FEATURES]
    y = df['target']

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "arn:aws:sagemaker:ap-south-1:718036509811:mlflow-tracking-server/iris-sagemaker-tracking"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-sagemaker"))

    with mlflow.start_run():
        model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        eval_metric="mlogloss"
        )

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("subsample", 1.0)
        mlflow.log_param("colsample_bytree", 1.0)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("eval_metric", "mlogloss")
    
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))
        mlflow.log_metric("accuracy", float(acc))
        print(f"accuracy={acc:.4f}")

        os.makedirs(model_path, exist_ok=True)
        joblib.dump(model, os.path.join(model_path, "model.joblib"))
        print(f"[INFO] Saved model to {model_path}")

        sig = infer_signature(Xtr, model.predict(Xtr))
        mlflow.xgboost.log_model(model, artifact_path="xgb_model",
                                signature=sig,
                                input_example=Xtr.head(3))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str,
                        default="/opt/ml/input/data/train")
    parser.add_argument("--model-path", type=str, default="/opt/ml/model")
    parser.add_argument("--n-estimators", type=int, default=20)
    return parser.parse_args()

if __name__ == "__main__":
    a = parse_args()
    model_train(a.input_path, a.model_path, n_estimators=a.n_estimators)
