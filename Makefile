PYTHON  ?= python
PIP     ?= pip

.PHONY: install install-dev test coverage ci-local train bundle-validate

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

test:
	pytest tests/unit/ -v

coverage:
	pytest tests/unit/ --cov=src --cov-report=term-missing --cov-report=html

ci-local: install-dev
	pytest tests/unit/ \
		--cov=src \
		--cov-report=term-missing \
		--junitxml=test-results.xml \
		--cov-fail-under=70

train:
	$(PYTHON) -m src.ml.churn_train --mock --tracking-uri ./mlruns

bundle-validate:
	databricks bundle validate --target staging
