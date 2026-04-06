import torch.nn as nn

            
def init_weights_kaiming_he(m: nn.Module) -> None:
    """
    Kaiming He initialization for linear layers.
    Ideal for models using ReLU or GELU activations.
    Keep activation variance constant (~1.0) across layers; this is more critical than the actual values themselves.
    Gradient survival: Prevent degeneration during the first epoch (neither vanishing, nor exploding).
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)            
