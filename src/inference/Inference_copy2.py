import os, tarfile, joblib, pandas as pd, argparse

def inference(model_dir,model_filename,input_dir,input_filename,output_dir):
    # --- Load model ---
    model_path = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_path):  
        tar_path = os.path.join(model_dir, "model.tar.gz")
        if os.path.exists(tar_path):
            with tarfile.open(tar_path) as tar:
                tar.extractall(model_dir)
    model = joblib.load(model_path)

    # --- Load input data ---
    df = pd.read_csv(os.path.join(input_dir, input_filename))
    X = df.drop(columns=["target"], errors="ignore")
    df_out = X.copy()
    
    # --- Predict & save output ---
    df_out["prediction"] = model.predict(X)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "predictions.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="/opt/ml/processing/input")
    p.add_argument("--input-filename", default="processed.csv")
    p.add_argument("--model-dir", default="/opt/ml/processing/model")
    p.add_argument("--model-filename", default="model.joblib")
    p.add_argument("--output-dir", default="/opt/ml/processing/output")
    args=p.parse_args()
    inference(args.model_dir,args.model_filename,args.input_dir,args.input_filename,args.output_dir)
