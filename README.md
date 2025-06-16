# energy-based-representation-learning
Unsupervised learning of energy representations using MRIs.

## Setup

```bash
pip install -r requirements-comet.txt
pre-commit install
```

## Reproduce our results

If you wish to train a new model, you can connect to the HPC (if you have access) and run the following command:

```bash
bsub < path/to/bsubs/scrupts
```

If not, you can simply run the train.py script and modify the config file you want to use. We have used 2DMRI_config.yml as the main config file for our runs. Hence in your terminal run:

```bash
python src/train.py
```
To choose an alternative configuration file than the default 2DMRI, you have to alter it in the script. If you run into memory issues, you can lower the batch size, number of steps and components significantly.


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

To run clevr dataset on our train:

```bash
python src/train.py --config='src/config/clevr_config.yml'
```

To avoid src module not found error, run the following command:
```bash
export PYTHONPATH="."
```

When you have queued the job, there a few good-to-know commands:

```bash
bjobs 
bkill job_id
bqueues
```

First one is to see the status of your queued jobs, second is to kill a specific job, last is to see the general queue on the HPC. 