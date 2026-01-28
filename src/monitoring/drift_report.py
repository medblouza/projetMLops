import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

reference = pd.read_csv("data/reference/reference.csv")
current = pd.read_csv("data/monitoring/current.csv")

report = Report(
    metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ]
)

report.run(
    reference_data=reference,
    current_data=current
)

report.save_html("data/monitoring/report.html")

print("Evidently report generated")
