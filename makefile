# Makefile for common tasks

.PHONY: help venv install test run clean download

help:
	@echo "QuantumCore Nexus - Make Targets"
	@echo ""
	@echo "  venv      Create virtual environment"
	@echo "  install   Install dependencies"
	@echo "  test      Run tests"
	@echo "  run       Run application"
	@echo "  clean     Clean up"
	@echo "  download  Download external modules"

venv:
	python3 -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/macOS"
	@echo "  venv\Scripts\activate     # Windows"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/ -v

run:
	python main.py

clean:
	rm -rf venv
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	rm -f *.log
	rm -rf logs/*

download:
	python scripts/download_modules.py

all: venv install download
	@echo "Setup complete. Run 'make run' to start."