import torch
import logging
import lightning
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from source.training_routine import TrainingRoutine
from source.select_dataloaders import select_dataloaders
from source.initialize_logging import initialize_logging

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_training_pipeline(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    mlf_logger, checkpoints = initialize_logging(
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        mlflow_tracking_uri=cfg.environment.mlflow_local_tracking_uri)

    mlf_logger.log_hyperparams(params=cfg)

    trainer = lightning.Trainer(
        accelerator=cfg.environment.accelerator,
        max_epochs=cfg.training.nb_epochs,
        profiler=None,
        num_sanity_val_steps=0,
        callbacks=checkpoints,
        logger=mlf_logger)

    routine = TrainingRoutine(cfg=cfg)

    dataloader_training, dataloader_validation, dataloader_test = select_dataloaders(cfg=cfg)

    torch.set_float32_matmul_precision(cfg.environment.torch_matmul_precision)

    trainer.fit(
        model=routine,
        train_dataloaders=dataloader_training,
        val_dataloaders=[dataloader_validation, dataloader_test])

    checkpoint_path_for_test = checkpoints[0].best_model_path

    trainer.test(
        model=routine,
        dataloaders=dataloader_test,
        ckpt_path=checkpoint_path_for_test)

if __name__ == '__main__':
    run_training_pipeline()
