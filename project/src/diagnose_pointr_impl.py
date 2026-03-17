"""Quick implementation diagnostics for PoinTr.

This script runs short, model-only checks to catch implementation bugs before
starting a long training run.
"""

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch

from models.pointr import PoinTr
from models.pointr.utils import ChamferDistanceL1


@dataclass
class PoinTrConfig:
    trans_dim: int = 384
    knn_layer: int = 0
    num_pred: int = 16384
    num_query: int = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_input(batch_size: int, num_points: int, device: str) -> torch.Tensor:
    pts = torch.randn(batch_size, num_points, 3, device=device)
    pts = pts / pts.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return pts


def check_determinism(model: torch.nn.Module, x: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        c1, f1 = model(x)
        c2, f2 = model(x)
    return {
        "eval_max_abs_diff_coarse": float((c1 - c2).abs().max().item()),
        "eval_max_abs_diff_fine": float((f1 - f2).abs().max().item()),
    }


def check_permutation_sensitivity(
    model: torch.nn.Module,
    x: torch.Tensor,
    chamfer: ChamferDistanceL1,
) -> dict:
    model.eval()
    with torch.no_grad():
        _, f_ref = model(x)
        perm = torch.randperm(x.shape[1], device=x.device)
        x_perm = x[:, perm, :]
        _, f_perm = model(x_perm)
        diff = chamfer(f_ref, f_perm)
    return {"permute_output_chamfer_l1": float(diff.item())}


def check_backward_finite(model: torch.nn.Module, x: torch.Tensor) -> dict:
    model.train()
    target = x.detach()
    c, f = model(x)
    loss_fn = ChamferDistanceL1()
    loss = loss_fn(c, target) + loss_fn(f, target)
    loss.backward()

    all_finite = True
    max_grad = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        finite = torch.isfinite(p.grad).all().item()
        all_finite = all_finite and bool(finite)
        max_grad = max(max_grad, float(p.grad.detach().abs().max().item()))

    model.zero_grad(set_to_none=True)
    return {
        "backward_loss": float(loss.detach().item()),
        "grads_all_finite": bool(all_finite),
        "max_abs_grad": max_grad,
    }


def check_input_scale_stability(model: torch.nn.Module, x: torch.Tensor) -> dict:
    model.eval()
    scales = [0.01, 0.1, 1.0, 10.0]
    out = {}
    with torch.no_grad():
        for scale in scales:
            _, f = model(x * scale)
            out[f"finite_output_scale_{scale}"] = bool(torch.isfinite(f).all().item())
            out[f"max_abs_output_scale_{scale}"] = float(f.abs().max().item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick diagnostics for PoinTr implementation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    requested_device = args.device
    if requested_device == "cuda":
        # Favor stable kernel selection for diagnostics over max throughput.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _run_checks(device: str) -> dict:
        model = PoinTr(PoinTrConfig()).to(device)
        x = make_input(args.batch_size, args.num_points, device)
        chamfer = ChamferDistanceL1()

        local_results = {}
        local_results.update(check_determinism(model, x))
        local_results.update(check_permutation_sensitivity(model, x, chamfer))
        local_results.update(check_backward_finite(model, x))
        local_results.update(check_input_scale_stability(model, x))
        return local_results

    used_device = requested_device
    try:
        results = _run_checks(used_device)
    except RuntimeError as exc:
        message = str(exc)
        should_fallback = (
            requested_device == "cuda"
            and ("cuDNN" in message or "CUDNN" in message or "Unable to find a valid cuDNN algorithm" in message)
        )
        if not should_fallback:
            raise

        print(f"[WARN] CUDA diagnostics failed ({exc}). Falling back to CPU.")
        used_device = "cpu"
        results = _run_checks(used_device)

    print("=== PoinTr Implementation Diagnostics ===")
    print(f"device_used: {used_device}")
    for key in sorted(results.keys()):
        print(f"{key}: {results[key]}")

    print("=== Suggested thresholds ===")
    print("eval_max_abs_diff_* should be exactly 0.0 in eval mode")
    print("grads_all_finite should be True")
    print("permute_output_chamfer_l1 should be small (closer to 0 is better)")


if __name__ == "__main__":
    main()
