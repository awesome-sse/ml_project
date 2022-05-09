import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=".", config_name="train_config")
def set_data_config(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    set_data_config()
