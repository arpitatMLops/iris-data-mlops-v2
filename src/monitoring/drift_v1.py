import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
import argparse

reference_path = "../preprocessing/processed_local/processed.csv"
current_path   = "../inference/output_local/predictions.csv"
output_html    = "./drift_report.html"

def drift(ref_dir:str,ref_filename:str,curr_dir:str,curr_filename:str,output_dir:str):

reference_path = os.path.join(ref_dir,ref_filename)
current_path = os.path.join(curr_dir,curr_filename)
ref = pd.read_csv(reference_path)
cur = pd.read_csv(current_path)

cur_t = cur.rename(columns={"prediction": "target"})

report = Report(metrics=[
    DataQualityPreset(),
    DataDriftPreset(),
    TargetDriftPreset()
])

report.run(reference_data=ref, current_data=cur_t)
os.makrdirs(output_dir,exist_ok=True)
out_file=os.path.join(output_dir,"drift_report1.html")
report.save_html(out_file)

print("Drift report saved to:", out_file)

if __name__ = "__main__":
    parser = argparser.ArgumentParser()
    parser.add_argument("--ref-dir",default="/opt/ml/processing/reference")
    parser.add_argument("--ref-filename", default="processed.csv")

    parser.add_argument("--curr-dir", default="/opt/ml/processing/current")
    parser.add_argument("--curr-filename", default="predictions.csv")

    parser.add_argument("--output-dir", default="/opt/ml/processing/output")



