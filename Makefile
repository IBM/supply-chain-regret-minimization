.PHONY: test

test:
	python -m mypy scregmin
	python -m pytest --disable-pytest-warnings
