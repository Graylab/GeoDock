import hydra
import torch
import os
from omegaconf import DictConfig

#os.environ["WANDB_API_KEY"] = 'bb76c909bdc55c9510c7e47d567c2f6c8a30f369'
#os.environ["WANDB_MODE"] = "offline"


@hydra.main(config_path="config/", config_name="config.yaml")
def main(config: DictConfig):
    torch.manual_seed(0)

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from geodock.trainer.train import train
    from geodock.trainer import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
