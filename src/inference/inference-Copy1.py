import os, tarfile, joblib, pandas as pd, argparse
from snowflake.connector import connect
from snowflake.connector.pandas_tools import write_pandas
from datetime import datetime, timezone


def main(args):
    # --- Load model ---
    model_path = os.path.join(args.model_dir, args.model_filename)
    if not os.path.exists(model_path):  
        tar_path = os.path.join(args.model_dir, "model.tar.gz")
        if os.path.exists(tar_path):
            with tarfile.open(tar_path) as tar:
                tar.extractall(args.model_dir)
    model = joblib.load(model_path)

    # --- Load input data ---
    df = pd.read_csv(os.path.join(args.input_dir, args.input_filename))
    X = df.drop(columns=["target"], errors="ignore")
    df_out = X.copy()
    
    # --- Predict & save output ---
    df_out["prediction"] = model.predict(X)
    df_out["run_id"] = args.run_id
    df_out["scored_at"] = datetime.now(timezone.utc)
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "predictions.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")

    print("Connecting to Snowflake")
    conn = connect(
        account="xj17520",
        user=args.sf_user,
        password=args.sf_password,
        warehouse=args.sf_warehouse,
        database=args.sf_database,
        schema=args.sf_schema,
        host="xj17520.eu-north-1.aws.snowflakecomputing.com"
    )

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS SNOWFLAKE_LEARNING_DB.PUBLIC.INFERENCE_SCORES(
                ID STRING,
                prediction DOUBLE,
                run_id STRING,
                scored_at TIMESTAMP_TZ
            )
        """)

    # Upload predictions
    ok, _, nrows, _ = write_pandas(conn, df_out, "SNOWFLAKE_LEARNING_DB.PUBLIC.INFERENCE_SCORES", auto_create_table=False)
    print(f"Loaded {nrows} rows to Snowflake.")

    conn.close()
    print("Connection closed.")
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="/opt/ml/processing/input")
    p.add_argument("--input-filename", default="processed.csv")
    p.add_argument("--model-dir", default="/opt/ml/processing/model")
    p.add_argument("--model-filename", default="model.joblib")
    p.add_argument("--output-dir", default="/opt/ml/processing/output")
    p.add_argument("--sf-user")
    p.add_argument("--sf-password")
    p.add_argument("--sf-warehouse", default="COMPUTE_WH")
    p.add_argument("--sf-database", default="SNOWFLAKE_LEARNING_DB")
    p.add_argument("--sf-schema", default="PUBLIC")
    p.add_argument("--run_id", default="batch-run")
    main(p.parse_args())
