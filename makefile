
install:
	pip install -r requirements-comet.txt

lint:
	pre-commit run --files src/**/*
