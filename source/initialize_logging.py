from omegaconf import DictConfig
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback


def initialize_logging(
    experiment_name: str,
    run_name: str,
    mlflow_tracking_uri: str) -> tuple[MLFlowLogger, list[Callback]]:
    """
    Initialize logger and checkpoint for MLFlow Tracking.

    Args:
        experiment_name: Name of the experiment.
        run_name: Name of the run.
        mlflow_tracking_uri: Uniform Resource Identifier for MLFlow Tracking.

    Returns:
        Logger and checkpoints.
    """

    CHECKPOINTS_FILENAME = "checkpoints_min_validation_loss"

    logger = MLFlowLogger(
        log_model="all",
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=mlflow_tracking_uri,
        synchronous=False)

    checkpoints = [
        ModelCheckpoint(save_top_k=1, monitor="validation_loss", mode="min", filename=CHECKPOINTS_FILENAME)]

    return logger, checkpoints
