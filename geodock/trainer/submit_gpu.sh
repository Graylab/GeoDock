#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=a100,v100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=72:00:00
#SBATCH --mem=40GB

#### execute code
python run.py model=geodock datamodule=geodock_datamodule logger.wandb.project=geodock trainer.max_epochs=30 callbacks.model_checkpoint.save_top_k=-1
