from lightning import LightningModule
from omegaconf import DictConfig
import torch

from source.select_loss import select_loss
from source.select_accuracy import select_accuracy
from source.select_optimizer import select_optimizer
from source.select_model import select_model


class TrainingRoutine(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.loss_function = select_loss(cfg=cfg)
        self.accuracy = select_accuracy(cfg=cfg)
        self.model = select_model(cfg=cfg)


    def forward(self, x):
        return self.model(input_images=x)


    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self.forward(x=images)

        if labels.shape != predictions.shape:
            # Enforce shape alignment
            # For classification: [B] -> [B, 1]
            # For segmentation: [B, H, W] -> [B, 1, H, W]
            labels = labels.view_as(predictions)

        # Loss: Expects Floats
        loss = self.loss_function(input=predictions, target=labels.float())

        if predictions.shape[1] == 1:
            # Predictions for Binary (C=1): Thresholding
            prediction_probabilities = torch.sigmoid(predictions)
            predictions_indices = (prediction_probabilities > 0.5).long()
        else:
            # Predictions for Multi-Class (C>1): Argmax
            predictions_indices = torch.argmax(predictions, dim=1)

        # Accuracy: Expects Longs
        accuracy = self.accuracy(predictions_indices.long(), labels.long())

        self.log(name="training_loss", value=loss, batch_size=images.shape[0], on_step=False, on_epoch=True)
        self.log('training_accuracy', accuracy, on_step=False, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx):
        _, labels = batch
        predictions = self.predict_step(batch, batch_idx)

        if labels.shape != predictions.shape:
            # Enforce shape alignment
            # For classification: [B] -> [B, 1]
            # For segmentation: [B, H, W] -> [B, 1, H, W]
            labels = labels.view_as(predictions)

        loss = self.loss_function(predictions, labels.float())

        if dataloader_idx == 0:
            self.log(
                name="validation_loss",
                value=loss,
                add_dataloader_idx=False,
                batch_size=labels.shape[0],
                on_step=False,
                on_epoch=True
            )
        else:
            self.log(
                name="test_loss",
                value=loss,
                add_dataloader_idx=False,
                batch_size=labels.shape[0],
                on_step=False,
                on_epoch=True
            )


    def test_step(self, batch, batch_idx):
        _, labels = batch
        predictions = self.predict_step(batch, batch_idx)

        if labels.shape != predictions.shape:
            # Enforce shape alignment
            # For classification: [B] -> [B, 1]
            # For segmentation: [B, H, W] -> [B, 1, H, W]
            labels = labels.view_as(predictions)

        loss = self.loss_function(input=predictions, target=labels.float())
        self.log(name="test_step_loss", value=loss, batch_size=labels.shape[0])


    def predict_step(self, batch, batch_idx):
        images, _ = batch
        predictions = self.forward(x=images)
        return predictions


    def configure_optimizers(self):

        optimizer = select_optimizer(cfg=self.cfg, parameters=self.parameters())

        # Warm-up scheduler: linear increase of the LR, from start_factor*lr to end_factor*lr during the warmup period
        warmup_period = self.cfg.training.get("warmup_epochs", 10)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.cfg.training.get("warmup_start_factor", 0.01),
            end_factor=self.cfg.training.get("warmup_end_factor", 1.0),
            total_iters=warmup_period
        )

        # Main scheduler: step LR decay
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.cfg.training.get("scheduler_step_size", 1),
            gamma=self.cfg.training.get("scheduler_gamma", 0.99)
        )

        # Chaining both schedulers together (indicating at which epoch to switch from "warmup" to "main")
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_period]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
