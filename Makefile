.PHONY: all ingest eda features experiments interpret cluster recommend clean help

PYTHON = ./venv/bin/python

help:
	@echo "Available commands:"
	@echo "  make all         - Run the complete research pipeline end-to-end"
	@echo "  make ingest      - Run ingestion, cleaning, track feature join & bias audit"
	@echo "  make eda         - Run exploratory data analysis and generate EDA figures"
	@echo "  make features    - Run strict causal feature engineering"
	@echo "  make experiments - Run model ladder and experiments A-E"
	@echo "  make interpret   - Run permutation importance, PDPs and case studies"
	@echo "  make cluster     - Run Add-On 1: Listening modes clustering"
	@echo "  make recommend   - Run Add-On 2: Offline skip-aware recommender"
	@echo "  make clean       - Remove generated parquet caches and pycache"

all:
	$(PYTHON) main.py --step all

ingest:
	$(PYTHON) main.py --step ingest

eda:
	$(PYTHON) main.py --step eda

features:
	$(PYTHON) main.py --step features

experiments:
	$(PYTHON) main.py --step experiments

interpret:
	$(PYTHON) main.py --step interpret

cluster:
	$(PYTHON) main.py --step cluster

recommend:
	$(PYTHON) main.py --step recommend

clean:
	rm -rf data/*.parquet
	find . -type d -name "__pycache__" -exec rm -rf {} +
