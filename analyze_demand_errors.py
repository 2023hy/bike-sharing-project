"""Recompute demand-stratified errors and the largest underprediction case."""
import csv
import json
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parent / "results"


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(name, rows):
    with (RESULTS / name).open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = json.loads((RESULTS / "model_selection.json").read_text())
    threshold = config["high_demand_threshold"]
    selected = config["regression_selected"]
    predictions = read_csv("regression_predictions.csv")
    summary = []
    for model in [selected, "WeeklySeasonal"]:
        rows = [r for r in predictions if r["model"] == model]
        for group, high in [("high", True), ("ordinary", False)]:
            errors = [float(r["prediction"]) - float(r["actual"]) for r in rows
                      if (float(r["actual"]) >= threshold) == high]
            summary.append(dict(model=model, group=group, threshold=threshold, hours=len(errors),
                                mae=mean(map(abs, errors)), mean_signed_error=mean(errors),
                                underprediction_fraction=mean(e < 0 for e in errors)))
    write_csv("demand_group_errors.csv", summary)
    worst = min((r for r in predictions if r["model"] == selected),
                key=lambda r: float(r["prediction"]) - float(r["actual"]))
    day = worst["timestamp"][:10]
    write_csv("largest_underprediction_day.csv",
              [r for r in predictions if r["model"] in (selected, "WeeklySeasonal")
               and r["timestamp"].startswith(day)])
    print("Largest underprediction:", worst)


if __name__ == "__main__":
    main()
