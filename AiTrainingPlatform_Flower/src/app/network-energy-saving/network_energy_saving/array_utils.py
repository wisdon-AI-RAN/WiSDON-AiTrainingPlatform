"""Helper utilities to pack and unpack multiple models for Flower."""

from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import torch
from flwr.app import ArrayRecord
from torch.nn import Module

_PREFIX_SEPARATOR = "/"


def pack_model_arrays(actor: Module, critic: Module) -> ArrayRecord:
    """Combine actor and critic weights into one ArrayRecord."""
    combined = OrderedDict()
    for name, tensor in actor.state_dict().items():
        combined[f"actor{_PREFIX_SEPARATOR}{name}"] = tensor.detach().cpu()
    for name, tensor in critic.state_dict().items():
        combined[f"critic{_PREFIX_SEPARATOR}{name}"] = tensor.detach().cpu()
    return ArrayRecord(combined)


def unpack_model_arrays(arrays: ArrayRecord) -> Tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    """Split a combined ArrayRecord back into actor and critic state dicts."""
    actor_sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    critic_sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    torch_state = arrays.to_torch_state_dict()
    for scoped_name, tensor in torch_state.items():
        try:
            scope, param_name = scoped_name.split(_PREFIX_SEPARATOR, 1)
        except ValueError as err:
            raise ValueError(f"Unexpected parameter name '{scoped_name}' in ArrayRecord") from err
        if scope == "actor":
            actor_sd[param_name] = tensor
        elif scope == "critic":
            critic_sd[param_name] = tensor
        else:
            raise ValueError(f"Unsupported scope '{scope}' in ArrayRecord")
    return actor_sd, critic_sd
