from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pytorch3d.ops import sample_farthest_points

from core import ModelConfig, create_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test for PointMAECompletion integration (forward/loss/backward)."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-input", type=int, default=2048)
    parser.add_argument("--num-target", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pointmae-ckpt",
        type=str,
        default=None,
        help="Optional upstream Point-MAE checkpoint for partial encoder init.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser


def _resolve_device(raw: str) -> torch.device:
    requested = raw.strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _make_synthetic_completion_batch(
    *,
    batch_size: int,
    num_input: int,
    num_target: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # We simulate complete objects and then downsample to create incomplete inputs.
    g = torch.Generator(device=device if device.type == "cuda" else "cpu")
    g.manual_seed(int(seed))

    full = torch.randn(
        batch_size, max(num_target, num_input), 3, device=device, generator=g
    )

    # Normalize each cloud to unit sphere for stable synthetic behavior.
    centroid = full.mean(dim=1, keepdim=True)
    full = full - centroid
    scale = full.norm(dim=2).amax(dim=1, keepdim=True).clamp_min(1e-6)
    full = full / scale.unsqueeze(-1)

    target, _ = sample_farthest_points(full, K=num_target)

    # Build incomplete input by taking a random view-like half-space crop + FPS.
    direction = torch.randn(batch_size, 1, 3, device=device, generator=g)
    direction = direction / direction.norm(dim=2, keepdim=True).clamp_min(1e-6)
    proj = (target * direction).sum(dim=2)
    keep_mask = proj >= proj.median(dim=1, keepdim=True).values

    padded_incomplete = []
    for b in range(batch_size):
        pts = target[b][keep_mask[b]]
        if pts.shape[0] < num_input:
            extra_idx = torch.randint(
                low=0,
                high=max(int(pts.shape[0]), 1),
                size=(num_input - pts.shape[0],),
                device=device,
                generator=g,
            )
            if pts.shape[0] == 0:
                pts = torch.zeros(1, 3, device=device)
            pts = torch.cat([pts, pts[extra_idx]], dim=0)
        pts = pts.unsqueeze(0)
        pts, _ = sample_farthest_points(pts, K=num_input)
        padded_incomplete.append(pts[0])

    incomplete = torch.stack(padded_incomplete, dim=0)
    return incomplete, target


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    device = _resolve_device(args.device)
    torch.manual_seed(int(args.seed))

    model_params = {
        "trans_dim": 384,
        "num_pred": 16384,
        "num_query": 224,
        "num_group": 128,
        "group_size": 32,
        "encoder_dims": 384,
        "depth": 8,
        "num_heads": 6,
        "decoder_depth": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "pointmae_ckpt": args.pointmae_ckpt,
    }

    print("[STEP 1] Creating model pointmae_completion...")
    model = create_model(
        ModelConfig(name="pointmae_completion", params=model_params),
        device=device,
    )
    model.train()

    print("[STEP 2] Building synthetic incomplete/target batch...")
    incomplete, target = _make_synthetic_completion_batch(
        batch_size=int(args.batch_size),
        num_input=int(args.num_input),
        num_target=int(args.num_target),
        device=device,
        seed=int(args.seed),
    )
    print(f"  incomplete shape: {tuple(incomplete.shape)}")
    print(f"  target shape:     {tuple(target.shape)}")

    print("[STEP 3] Forward pass...")
    pred = model(incomplete)
    if not isinstance(pred, (tuple, list)) or len(pred) != 2:
        raise RuntimeError("Expected model output to be tuple(coarse, fine).")

    coarse, fine = pred
    print(f"  coarse shape:     {tuple(coarse.shape)}")
    print(f"  fine shape:       {tuple(fine.shape)}")

    print("[STEP 4] Loss computation...")
    core_model = model.module if hasattr(model, "module") else model
    loss_coarse, loss_fine = core_model.get_loss(pred, target)
    total_loss = loss_coarse + loss_fine
    print(f"  loss coarse:      {float(loss_coarse.detach().item()):.6f}")
    print(f"  loss fine:        {float(loss_fine.detach().item()):.6f}")
    print(f"  loss total:       {float(total_loss.detach().item()):.6f}")

    if not torch.isfinite(total_loss):
        raise RuntimeError("Total loss is not finite.")

    print("[STEP 5] Backward + optimizer step...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=1e-4
    )
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f"  grad norm (clipped): {float(grad_norm):.6f}")

    print("[OK] test3 smoke test finished successfully.")


if __name__ == "__main__":
    main()
