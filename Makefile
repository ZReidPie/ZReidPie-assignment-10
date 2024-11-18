# Define your virtual environment and flask app
VENV = venv
FLASK_APP = app.py

# Detect the operating system
ifeq ($(OS),Windows_NT)
    PIP = .\$(VENV)\Scripts\pip
    FLASK = .\$(VENV)\Scripts\flask
else
    PIP = ./$(VENV)/bin/pip
    FLASK = ./$(VENV)/bin/flask
endif

# Install dependencies
install:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Run the Flask application
# Run the Flask application
run:
ifeq ($(OS),Windows_NT)
	set FLASK_APP=$(FLASK_APP) && set FLASK_ENV=development && $(FLASK) run --port 3000
else
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development $(FLASK) run --port 3000
endif


# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install