# =============================================================================
# Chest X-ray Benchmark — Makefile
# =============================================================================

.PHONY: install dataset features train-classical train-cnn train-transformer \
        train-detection evaluate explain quantise api clean

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	pip install -e .

# ── Pipeline Steps ────────────────────────────────────────────────────────────
dataset:
	python scripts/01_build_dataset.py

features:
	python scripts/02_extract_features.py

train-classical:
	python scripts/03_train_classical.py

train-cnn:
	python scripts/04_train_cnn.py

train-transformer:
	python scripts/05_train_transformer.py

prepare-detection:
	python scripts/06_prepare_detection.py

train-yolo:
	python scripts/07_train_yolo.py

train-detection:
	python scripts/08_train_detection.py

evaluate:
	python scripts/09_evaluate_all.py

explain:
	python scripts/10_generate_explanations.py

quantise:
	python scripts/11_quantise_export.py

# ── API ───────────────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker-compose -f api/docker-compose.yml build

docker-up:
	docker-compose -f api/docker-compose.yml up -d

# ── Full Pipeline ─────────────────────────────────────────────────────────────
all: dataset features train-classical train-cnn train-transformer evaluate explain

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
