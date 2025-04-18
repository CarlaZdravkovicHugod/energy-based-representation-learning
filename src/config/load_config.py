from dataclasses import dataclass
import os
import torch
import yaml
from dotenv import load_dotenv

from src.utils.neptune_logger import NeptuneLogger

load_dotenv()

@dataclass
class Config:
    run_name: str
    test_run: bool
    batch_size: int
    data_workers: int # TODO: determine
    shuffle_dataset: bool
    num_epoch: int # TODO: determine
    lr: float
    step_lr: float # TODO: determine
    components: int # TODO: determine
    ensembles: int # not sure what this is :/
    data_path: str
    train_data_size: float # TODO: determine
    filter_dim: int # number of filters to use
    latent_dim: int # latent dimension
    pos_embed: bool # use positional embedding
    recurrent_model: bool # use a recurrent model to infer the latents
    dataset: str # dataset to use
    steps: int # TODO: determine optimal
    decoder: bool
    channels: int # number of channels in the input image
    model: str # model to use, either 'LatentEBM' or 'LatentEBM128'
    
    def to_dict(self):
        copy = self.__dict__.copy()
        copy["NeptuneLogger"] = None
        return copy


def logic_check(config_dict: dict):
    if os.environ["NEPTUNE_API_TOKEN"] is None:
        config_dict["test_run"] = True
        print("No NEPTUNE_API_TOKEN found, setting test_run to True = Metrics only locally")
    return config_dict

def load_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = logic_check(config_dict)

    return Config(
        **config
    )

if __name__ == "__main__":
    config = load_config("src/config/test.yml")
    print(config)