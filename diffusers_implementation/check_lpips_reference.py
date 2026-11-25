"""Check local LPIPS calculation against upstream reference implementation."""
import importlib.util
import sys
import types
from pathlib import Path

import torch

from experiment_lpips_content import LPIPSAlex


def main():
    root = Path("downloads/PerceptualSimilarity/lpips").resolve()
    # Load the upstream metric and backbone without importing its unused
    # training module, which requires scipy.
    package = types.ModuleType("lpips")
    package.__path__ = [str(root)]
    package.normalize_tensor = lambda x: x / (torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-10)
    sys.modules["lpips"] = package
    spec = importlib.util.spec_from_file_location("lpips.lpips", root / "lpips.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lpips.lpips"] = module
    spec.loader.exec_module(module)
    reference = module.LPIPS(pretrained=False, net="alex", pnet_rand=True, verbose=False).eval()
    model = LPIPSAlex("downloads/torchvision/alexnet-owt-7be5be79.pth", root / "weights/v0.1/alex.pth").eval()
    backbone = torch.load("downloads/torchvision/alexnet-owt-7be5be79.pth", map_location="cpu", weights_only=True)
    for name, tensor in reference.net.state_dict().items():
        layer_name = name.split(".")[1:]
        tensor.copy_(backbone["features." + ".".join(layer_name)])
    head = torch.load(root / "weights/v0.1/alex.pth", map_location="cpu", weights_only=True)
    reference.load_state_dict(head, strict=False)
    generator = torch.Generator().manual_seed(19)
    x = torch.rand((1, 3, 64, 64), generator=generator) * 2 - 1
    y = torch.rand((1, 3, 64, 64), generator=generator) * 2 - 1
    with torch.inference_mode():
        actual = model.distance(model.extract(x), model.extract(y)).item()
        expected = reference(x, y).item()
        identical = model.distance(model.extract(x), model.extract(x)).item()
    print(f"Reference LPIPS={expected:.9f}, implementation={actual:.9f}, self={identical:.9f}")
    if abs(actual - expected) > 1e-6 or abs(identical) > 1e-8:
        raise AssertionError("LPIPS mismatch with official implementation")


if __name__ == "__main__":
    main()
