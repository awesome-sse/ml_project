import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_name="data_config")
def set_data_config(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    set_data_config()