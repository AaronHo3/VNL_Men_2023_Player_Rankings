PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

.PHONY: setup install download-data cluster cluster-tsne rank notebook test clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

install:
	pip install -r requirements.txt

download-data:
	$(VENV_PYTHON) scripts/download_kaggle_data.py

cluster:
	$(VENV_PYTHON) scripts/run_player_clustering.py

cluster-tsne:
	$(VENV_PYTHON) scripts/run_player_clustering.py --with-tsne

rank:
	$(VENV_PYTHON) scripts/run_player_ranking.py

notebook:
	$(VENV_PYTHON) -m notebook

test:
	$(VENV_PYTHON) -m pytest -q

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
