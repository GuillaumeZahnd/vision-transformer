import torch
import time
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from source.training_routine import TrainingRoutine
from source.select_dataloaders import select_dataloaders


@hydra.main(version_base=None, config_path="config", config_name="config")
def benchmark_inference_latency(cfg: DictConfig):

    logging.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.environment.accelerator == "gpu" else "cpu")
    logging.info(f"Benchmarking on device: {device}")

    # Model
    model = TrainingRoutine(cfg=cfg)
    model.to(device)
    model.eval()

    # Enforce a unit batch size to measure single-sample latency
    OmegaConf.update(cfg, "training.batch_size", 1, force_add=True)

    # Data
    _, _, dataloader_test = select_dataloaders(cfg=cfg)

    # Consider only a reduced amount of samples for the benchmark
    nb_samples = 100
    nb_steps_warmup = 10

    latencies = []

    logging.info(f"Starting benchmark with {nb_steps_warmup} warmup steps and {nb_samples} measured samples.")

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(dataloader_test)):

            inputs = batch[0].to(device)
            batch_size = inputs.size(0)

            # Warmup
            if batch_index < nb_steps_warmup:
                _ = model(inputs)
                continue

            # Exit benchmark after measuring enough samples
            if batch_index >= (nb_samples + nb_steps_warmup):
                break

            # Synchronize CUDA for accurate timing on GPU
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            # This is the critical forward pass operation that we want to benchmark
            _ = model(inputs)

            # Synchronize CUDA for accurate timing on GPU
            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Per-sample time for this batch (for good practice, we have already ensured batch_size=1)
            time_per_sample = (end_time - start_time) / batch_size
            latencies.append(time_per_sample)

    # Results
    average_latency_ms = 1000*sum(latencies) / len(latencies)
    frame_rate = 1000 / average_latency_ms

    logging.info(f"Average time per sample: {average_latency_ms:.2f}ms.")
    logging.info(f"Frame rate: {frame_rate:.2f}Hz.")


if __name__ == "__main__":
    benchmark_inference_latency()
