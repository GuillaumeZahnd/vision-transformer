import torch
import numpy as np
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid, draw_segmentation_masks


class SegmentationLoggerCallback(Callback):
    def __init__(self, nb_examples: int):
        super().__init__()
        self.nb_examples = nb_examples


    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.eval()

        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        batch = next(iter(val_loader))
        images, labels = batch[0][:self.nb_examples], batch[1][:self.nb_examples]

        images = images.to(pl_module.device)
        with torch.no_grad():
            logits = pl_module(images)
            if logits.shape[1] == 1:
                # Binary case: Threshold the sigmoid
                predictions = (torch.sigmoid(logits) > 0.5).squeeze(1)
                nb_classes = 2
            else:
                # Multi-class: Argmax
                predictions = torch.argmax(logits, dim=1)
                nb_classes = logits.shape[1]

        # Enfore three channels because draw_segmentation_masks() requires an RGB image
        images = images.cpu()
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Cast images to uint8 [0, 255] to call draw_segmentation_masks()
        images = (images * 255).clamp(0, 255).to(torch.uint8)

        labels = labels.cpu().long()
        predictions = predictions.cpu().long()
        if labels.ndim == 4:
            labels = labels.squeeze(1)

        nb_classes = logits.shape[1]
        overlays_ground_truth = []
        overlays_predictions = []

        for i in range(images.shape[0]):
            # Create boolean masks for each class, of shape (nb_classes, H, W)
            masks_ground_truth = torch.stack([labels[i] == c for c in range(nb_classes)])
            masks_prediction = torch.stack([predictions[i] == c for c in range(nb_classes)])

            # Draw overlays (alpha=0.5 for transparency)
            masks_ground_truth_overlay = draw_segmentation_masks(images[i], masks_ground_truth, alpha=0.5)
            masks_prediction_overlay = draw_segmentation_masks(images[i], masks_prediction, alpha=0.5)

            overlays_ground_truth.append(masks_ground_truth_overlay)
            overlays_predictions.append(masks_prediction_overlay)

        # Convert lists back to tensors [B, 3, H, W]
        overlays_ground_truth = torch.stack(overlays_ground_truth)
        overlays_predictions = torch.stack(overlays_predictions)

        # Concatenate along the width dimension
        combined = torch.cat([images, overlays_ground_truth, overlays_predictions], dim=3)

        # Creates a grid with nb_examples cases along the vertical axis
        grid = make_grid(combined, nrow=1, padding=5)

        # Convert NumPy array, of shape (H, W, C)
        image_array = grid.permute(1, 2, 0).numpy()

        trainer.logger.experiment.log_image(
            run_id=trainer.logger.run_id,
            image=image_array,
            artifact_file=f"validation_visuals/epoch_{trainer.current_epoch}.png"
        )
