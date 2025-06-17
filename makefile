
install:
	pip install -r requirements.txt

lint:
	pre-commit run --files src/**/*

train:
	python src/train.py --config src/config/2DMRI_config.yaml
