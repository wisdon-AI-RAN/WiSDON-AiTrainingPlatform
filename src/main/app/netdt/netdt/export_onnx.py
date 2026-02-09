import sys, os
import torch
import torch.nn as nn
from netdt.model import AERegressor

def export_onnx(model, onnx_path, x_mu, x_std, y_mu, y_std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = AEInferenceWrapper(model, x_mu, x_std, y_mu, y_std).to(device)
    wrapper.eval()
    dummy = torch.randn(1, 4, device=device, dtype=torch.float32)
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            dynamo=False,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["x_raw"],
            output_names=["pred_rsrp_dbm"],
            dynamic_axes={"x_raw": {0: "batch"}, "pred_rsrp_dbm": {0: "batch"}},
        )
        print(f"[OK] Saved ONNX model: {onnx_path}")
    except Exception as e:
        print(f"[WARN] ONNX export skipped: {e}")

class AEInferenceWrapper(nn.Module):
    """
    ONNX export wrapper: input raw features [sim_rsrp, pci, x, y], output predicted RSRP in dB.
    """

    def __init__(self, base_model: AERegressor, x_mu, x_std, y_mu, y_std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("x_mu", torch.tensor(x_mu, dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor(x_std, dtype=torch.float32))
        self.register_buffer("y_mu", torch.tensor(y_mu, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x_n = (x_raw - self.x_mu) / self.x_std
        y_n, _ = self.base_model(x_n)
        y_db = y_n * self.y_std + self.y_mu
        return y_db