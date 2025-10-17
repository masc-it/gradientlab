from torch import nn

from gradientlab.neuralblocks.norm_layers.rmsnorm import RMSNorm


def get_adamw_parameters(model: nn.Module, weight_decay: float):
    """
    Returns parameter groups for the AdamW optimizer from the given model.

    This function splits the parameters into two groups:
      1. Parameters that will have weight decay applied.
      2. Parameters that will not have weight decay (e.g. biases and LayerNorm weights).

    Args:
        model (nn.Module): The neural network model.
        weight_decay (float): The weight decay coefficient to apply to applicable parameters.

    Returns:
        list[dict]: A list containing two dictionaries, each with a 'params' key and a
                    'weight_decay' key. This list is ready to be passed to AdamW.
    """
    decay, no_decay = [], []
    norm_types = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.Embedding,
        nn.RMSNorm,
        RMSNorm,
    )

    # 1) Collect all param objects inside normalisation layers
    norm_params = {
        p
        for m in model.modules()
        if isinstance(m, norm_types)
        for p in m.parameters(recurse=False)
    }

    # 2) Split
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p in norm_params or p.ndim == 1:  # 1-D covers biases & LayerNorm weights
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
