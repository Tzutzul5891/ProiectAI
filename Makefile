.PHONY: setup model run

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

model:
	. .venv/bin/activate && python scripts/get_model.py

run:
	. .venv/bin/activate && python script1.py
