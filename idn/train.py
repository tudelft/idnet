from omegaconf import OmegaConf
import hydra
from .utils.trainer import Trainer

# @hydra.main(config_path="config", config_name="mvsec_train")
# @hydra.main(config_path="config", config_name="tid_train")
@hydra.main(config_path="config", config_name="id_train")

def main(config):
    print(OmegaConf.to_yaml(config))

    trainer = Trainer(config)

    print("Number of parameters: ", sum(p.numel()
          for p in trainer.model.parameters() if p.requires_grad))

    trainer.fit()


if __name__ == '__main__':
    main()
