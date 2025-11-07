import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset

reference_path = "../preprocessing/processed_local/processed.csv"
current_path   = "../inference/output_local/predictions.csv"
output_html    = "./drift_report.html"

ref = pd.read_csv(reference_path)
cur = pd.read_csv(current_path)

cur_t = cur.rename(columns={"prediction": "target"})

report = Report(metrics=[
    DataQualityPreset(),
    DataDriftPreset(),
    TargetDriftPreset()
])

report.run(reference_data=ref, current_data=cur_t)
report.save_html(output_html)

print("Drift report saved to:", output_html)
