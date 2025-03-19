# energy-based-representation-learning
Unsupervised learning of energy representations using MRIs.

## Setup

```bash
pip install -r requirements-comet.txt
pre-commit install
```

## Usage


If you want to run the comet code, go to the comet dir ''cd comet'' and run the following command:

Their clevr dataset:
```bash
python train.py --exp=clevr --batch_size=12 --gpus=0 --cuda --train --dataset=clevr --step_lr=500.0
```

Our Data: 
```bash
python train.py --exp=MRI2D --batch_size=12 --gpus=0 --cuda --train --dataset=MRI --step_lr=500.0
```

To run clevr code on our train:

```bash
python src/train.py --config='src/config/clevr_config.yml'
```