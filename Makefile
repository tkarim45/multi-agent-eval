PY ?= ~/miniconda3/envs/personal/bin/python
PIP ?= ~/miniconda3/envs/personal/bin/pip
.PHONY: install bench live test
install:
	$(PIP) install -e ".[all]"
bench:
	$(PY) -m maeval.harness
live:
	$(PY) -m maeval.harness --live
test:
	$(PY) -m pytest -q
