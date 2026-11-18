"""Promocja modelu Staging -> Production w MLflow Registry."""
from __future__ import annotations

import sys

import mlflow

MODEL_NAME = "datamart-pro-churn-model"


def main() -> None:
    client   = mlflow.tracking.MlflowClient()
    staging  = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not staging:
        print("Brak modelu w Staging — nic nie promuje")
        sys.exit(1)

    v = staging[0]
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=v.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted {MODEL_NAME} v{v.version} -> Production")


if __name__ == "__main__":
    main()
