"""Gate check — blokuje deployment jeśli model Staging ma AUC < 0.75."""
from __future__ import annotations

import sys

import mlflow

THRESHOLD = 0.75
MODEL_NAME = "datamart-pro-churn-model"


def main() -> None:
    client   = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not versions:
        print("Brak modelu w Staging")
        sys.exit(1)

    auc = client.get_run(versions[0].run_id).data.metrics.get("roc_auc", 0.0)
    print(f"AUC = {auc:.4f} (prog: {THRESHOLD})")

    if auc < THRESHOLD:
        print("Nie spelnia progu — deployment zablokowany")
        sys.exit(1)

    print("Prog spelniony")


if __name__ == "__main__":
    main()
