# =========================================================================
# LLM-Book-Summarizer – local developer helper
#
#   make setup     → create ./venv and install pinned dependencies
#   make run       → start the Streamlit front-end using the venv
#   make clean     → remove the venv and Python caches
# -------------------------------------------------------------------------

# ---- Tunables -----------------------------------------------------------
PY          ?= python3               # override with PY=<path/to/python>
VENV        ?= venv                 # directory that will hold the venv
REQ_FILE   ?= requirements.txt      # dependency list

PIP         := $(VENV)/bin/pip
ACTIVATE    := . $(VENV)/bin/activate

# ---- Helper targets -----------------------------------------------------

$(VENV)/bin/python:
	@echo "[+] Creating virtual environment in '$(VENV)'"
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: setup
setup: $(VENV)/bin/python
	@echo "[+] Installing Python dependencies"
	$(PIP) install -r $(REQ_FILE)
	@echo "[✓] Environment ready – activate with 'source $(VENV)/bin/activate'"

.PHONY: run
run: setup
	@echo "[»] Launching Streamlit…"
	$(VENV)/bin/streamlit run app.py

.PHONY: clean
clean:
	rm -rf $(VENV) __pycache__ */__pycache__ .pytest_cache
	@echo "[✓] Cleaned" 