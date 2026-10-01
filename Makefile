PYTHON ?= python3

.PHONY: check test

check:
	$(PYTHON) -m compileall -q train.py predict.py src tests bonus

test:
	$(PYTHON) -m unittest discover -s tests -v
