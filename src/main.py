import os
import argparse
from src.preprocessing.preprocessing import run_preprocessing
from src.model_training.sagemaker_train import model_train
from src.inference.inference import main as run_inference 

def main():
    step = os.getenv("STEP", "train").lower()

    if step == "preprocess":
        # SageMaker Processing writes outputs here
        out_dir = os.getenv("OUTPUT_DIR", "/opt/ml/processing/output")
        print(f"STEP=preprocess → output_dir={out_dir}")
        run_preprocessing(out_dir)

    elif step == "train":
      
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-path", type=str,
                            default="/opt/ml/input/data/train")
        parser.add_argument("--model-path", type=str, default="/opt/ml/model")
        parser.add_argument("--n-estimators", type=int, default=20)
        args, _unknown = parser.parse_known_args()

        print(f"STEP=train → input={args.input_path} model_dir={args.model_path}")
        model_train(args.input_path, args.model_path, n_estimators=args.n_estimators)

    elif step == "infer":
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
        parser.add_argument("--input-filename", type=str, default="processed.csv")
        parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
        parser.add_argument("--model-filename", type=str, default="model.joblib")
        parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
        args, _ = parser.parse_known_args()

        print(f"STEP=infer → input={args.input_dir}/{args.input_filename}, "
              f"model={args.model_dir}/{args.model_filename}, output={args.output_dir}")
        run_inference(args)

    else:
        raise ValueError(f"Unknown STEP={step}. Use preprocess | train | infer.")


if __name__ == "__main__":
    main()
