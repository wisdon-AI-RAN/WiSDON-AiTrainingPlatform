import sys, os
import inspect
import torch
import numpy as np
import torch.nn as nn

from network_energy_saving.model import Actor, DQN

def export_onnx(model: Actor, onnx_path: str, opset: int = 18, n_feats: int = 9, total_bs: int = 8) -> None:
    """
    Export deterministic actor logits model to ONNX.
    Input:  state  [B, N_FEATS, 10, TOTAL_BS]
    Output: logits [B, ACTION_DIM]
    """
    model.eval()

    class ActorLogitsWrapper(nn.Module):
        def __init__(self, actor: Actor):
            super().__init__()
            self.actor = actor
        def forward(self, state):
            return self.actor.forward_logits(state)

    orig_device = next(model.parameters()).device
    cpu_model = model.to("cpu").eval()
    wrapper = ActorLogitsWrapper(cpu_model).eval()

    dummy = torch.zeros(1, n_feats, 10, total_bs, dtype=torch.float32, device="cpu")

    export_kwargs = {
        "input_names": ["state"],
        "output_names": ["logits"],
        "dynamic_axes": {"state": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": opset,
    }
    if "use_dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["use_dynamo"] = False

    try:
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            **export_kwargs,
        )
    finally:
        model.to(orig_device)
    print(f"[EXPORT] ONNX saved → {onnx_path}")