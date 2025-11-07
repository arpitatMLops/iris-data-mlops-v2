from sklearn.datasets import load_iris
import pandas as pd
import os
import argparse

def run_preprocessing(output_dir: str):
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target"] = iris.target

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "processed.csv")
    df.to_csv(out_file, index=False)
    print("processed file created", out_file, " shape:", df.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                        default="/opt/ml/processing/output")
    args = parser.parse_args()
    run_preprocessing(args.output_dir)
