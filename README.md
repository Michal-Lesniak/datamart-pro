# DataMart Pro — MLOps Pipeline (Miesiąc 11)

[![CI](https://github.com/<org>/datamart-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/<org>/datamart-pro/actions/workflows/ci.yml)

Produkcyjny system predykcji churnu dla DataMart Pro:
- **CI/CD** w GitHub Actions (push + PR triggery, gate check, deploy staging/production).
- **MLflow** experiment tracking + Model Registry (3 modele, auto-promocja do Staging).
- **Testy jednostkowe** z coverage ≥ 80% dla `silver_transform` i `churn_features`.
- **Databricks Bundle** do deploymentu Spark jobów.

---

## Architektura DataMart Pro (M1–M11)

```
      ┌──────────────┐   Auto Loader   ┌──────────────┐
      │  Source apps │ ───────────────▶│   Bronze     │
      │  (M3 ADF/ADLS│                 │   (Delta)    │
      └──────────────┘                 └──────┬───────┘
                                              ▼
                                       ┌──────────────┐
                                       │  Silver      │  ◀── unit tests (M10, M11)
                                       │  silver_     │      + GE (M10)
                                       │  transform   │
                                       └──────┬───────┘
                                              ▼
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                       ┌──────────────┐             ┌──────────────┐
                       │  Gold        │             │  ML Features │
                       │  dashboards  │             │  (M11)       │
                       └──────────────┘             └──────┬───────┘
                                                            ▼
                                                     ┌──────────────┐
                                                     │  MLflow      │
                                                     │  Registry    │
                                                     │  Staging/Prod│
                                                     └──────────────┘
```

Warstwy historyczne:
- **M1-M3** — Python ETL + ADF/ADLS (ecommerce_etl, ecommerce_adf)
- **M4** — Airflow DAGs
- **M5-M6** — Databricks + dbt (ecommerce_dbt)
- **M7** — dbt tests & docs
- **M8** — ADF/Fabric
- **M9** — Spark Structured Streaming (nexashop_m09_streaming)
- **M10** — Data Quality + Observability (miesiac10_certyfikat)
- **M11** — CI/CD + MLflow (ten projekt)

---

## CI/CD — przepływ

```
developer push -> any branch
                       │
                       ▼
                  ┌──────────┐
                  │   CI     │   .github/workflows/ci.yml
                  │ pytest   │   coverage ≥ 70% (fail otherwise)
                  │ artifacts│   test-results.xml + coverage.xml
                  └────┬─────┘
                       │ merge to develop
                       ▼
                  ┌──────────┐
                  │ CD STG   │   cd_staging.yml
                  │ validate │   databricks bundle validate
                  │ deploy   │   bundle deploy --target staging
                  │ smoke    │   trigger silver_transform_job
                  └────┬─────┘
                       │ merge to main
                       ▼
                  ┌──────────┐
                  │ CD PROD  │   cd_production.yml
                  │ gate AUC │   AUC(Staging model) >= 0.75 → BONUS
                  │ review   │   GitHub Environment "production"
                  │ deploy   │   bundle deploy --target production
                  │ promote  │   MLflow Staging → Production
                  └──────────┘
```

---

## Uruchomienie lokalnie

### Wymagania
- Python 3.11
- Java 17 (dla PySpark)
- `make`

### Setup

```bash
make install-dev
```

### Testy + coverage

```bash
make ci-local
# równoważne:
# pytest tests/unit/ --cov=src --cov-report=term-missing --cov-fail-under=70
```

Oczekiwany wynik: 20+ testów przechodzi, coverage dla `silver_transform` i `churn_features` ≥ 80%.

### Trening modelu lokalnie

```bash
make train
# → mlflow ui       (osobny terminal, tracking URI = ./mlruns)
# otwórz http://127.0.0.1:5000 i obejrzyj 3 runy + model w Registry
```

### Weryfikacja Databricks Bundle (bez workspace'a)

```bash
databricks bundle validate --target staging
```

---

## Struktura repo

```
datamart-pro/
├── .github/workflows/
│   ├── ci.yml                    # push + PR → pytest + coverage
│   ├── cd_staging.yml            # push develop → deploy + smoke
│   └── cd_production.yml         # push main  → gate + manual approval + promocja
├── src/
│   ├── pipelines/silver_transform.py   # 5 funkcji z type hints + docstrings
│   └── ml/
│       ├── churn_features.py           # feature engineering (pandas)
│       ├── churn_train.py              # MLflow — 3 runy + rejestracja
│       └── churn_predict.py            # batch scoring
├── tests/
│   ├── conftest.py                # SparkSession scope=session
│   └── unit/
│       ├── test_transformations.py # 20 testów (parametrize + raises + empty)
│       └── test_features.py        # 15 testów (boundary + raises)
├── scripts/
│   ├── check_model_quality.py     # gate AUC ≥ 0.75
│   └── promote_model.py           # Staging → Production
├── databricks.yml
├── Makefile
├── requirements.txt / requirements-dev.txt
└── README.md
```

---

## MLflow — jak wygląda to w praktyce

Po uruchomieniu `make train`:

1. **Eksperyment** `datamart-pro-churn` z 3 runami (`logreg`, `rf`, `gbt`)
2. Każdy run loguje:
   - params (`model_type`, hiperparametry)
   - metrics (`roc_auc`, `f1`, `precision`, `recall`)
   - artefakty (`classification_report.txt`, `confusion_matrix.csv`)
   - dla modeli drzewiastych: `feature_importances.csv`
3. **Najlepszy model** (wg `roc_auc`) → zarejestrowany jako `datamart-pro-churn-model` w stage=**Staging**

Screenshoty do oddania:
- `screenshot_ci.png` — zielony check w GitHub Actions
- `screenshot_mlflow_runs.png` — lista 3 runów z metrykami
- `screenshot_mlflow_registry.png` — model w Registry ze stage=Staging

---

## BONUS — cd_production.yml

Pipeline produkcyjny:
1. Reużywa testów CI jako `needs:` gate.
2. Dodatkowy gate `scripts/check_model_quality.py` — blokuje deploy jeśli AUC < 0.75.
3. GitHub Environment `production` z **required reviewers** (konfiguracja w Settings → Environments).
4. Po udanym deploy — `scripts/promote_model.py` promuje model Staging → Production (z `archive_existing_versions=True`).

---

## Co testujemy, czego nie testujemy

**Testy jednostkowe** (`tests/unit/`):
- `silver_transform.py` — wszystkie publiczne funkcje, parametryzacja po VALID_STATUSES, boundary values, empty DF, wyjątki.
- `churn_features.py` — boundary cases (recency_days = 89/90/91), NULL handling, wyjątki dla złego progu.

**Czego nie testujemy jednostkowo** (celowo):
- `churn_train.py::train()` — integruje MLflow + sklearn + mock data; test integracyjny osobno (w CI jako job opcjonalny).
- `churn_predict.py::load_model` — wymaga MLflow registry; weryfikacja w smoke tests na staging.

---

## Jak zdobyć punkty bonusowe

- cd_production.yml z gate'em AUC — +10 pkt
- Reviewers w GitHub Environment "production"
- `promote_model.py` wywoływany dopiero po udanym `databricks bundle deploy`
