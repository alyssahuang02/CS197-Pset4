from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def run(cfg: DictConfig):
    opt = instantiate(cfg)
    print(opt)

if __name__ == "__main__":
    run()