from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from torch.utils.data import DataLoader, Dataset

from core import (
    bootstrap,
    create_advanced_reconstruction_dataset,
    create_basic_reconstruction_dataset,
    create_model,
    load_model_checkpoint,
)
from dataset.wrapper import PointcloudPatchDataset


class _SingleCloudDataset(Dataset):
    def __init__(self, defected: np.ndarray, original: np.ndarray):
        self.defected = np.asarray(defected, dtype=np.float32)
        self.original = np.asarray(original, dtype=np.float32)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        return {
            "defected_pos": torch.from_numpy(self.defected).float(),
            "original_pos": torch.from_numpy(self.original).float(),
        }


def _extract_cloud_pair(sample) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(sample, "defected_pos"):
        defected = sample.defected_pos
        original = getattr(sample, "original_pos", defected)
    elif isinstance(sample, dict):
        defected = sample.get("defected_pos", sample.get("pos", None))
        if defected is None:
            raise ValueError("Sample is missing defected_pos/pos")
        original = sample.get("original_pos", defected)
    else:
        raise TypeError(f"Unsupported sample type: {type(sample).__name__}")

    defected_np = np.asarray(
        torch.as_tensor(defected).float().cpu().numpy(), dtype=np.float32
    )
    original_np = np.asarray(
        torch.as_tensor(original).float().cpu().numpy(), dtype=np.float32
    )

    if defected_np.ndim != 2 or defected_np.shape[1] != 3:
        raise ValueError(f"defected cloud must be (N,3), got {defected_np.shape}")
    if original_np.ndim != 2 or original_np.shape[1] != 3:
        raise ValueError(f"original cloud must be (N,3), got {original_np.shape}")

    return defected_np, original_np


def _safe_first_patch_radius(value, fallback: float) -> float:
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return float(value[0])
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _downsample_points_to_fixed_count(
    points_np: np.ndarray,
    target_count: int,
    seed: int,
) -> np.ndarray:
    pts = np.asarray(points_np, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {pts.shape}")

    n = int(pts.shape[0])
    k = int(target_count)
    if k <= 0:
        raise ValueError("target_count must be > 0")

    pts_t = torch.from_numpy(pts).float().unsqueeze(0)

    if n >= k:
        sampled, _ = sample_farthest_points(
            pts_t,
            K=k,
            random_start_point=False,
        )
        return sampled[0].cpu().numpy().astype(np.float32)

    # If we have fewer points than requested, keep FPS over all points and pad.
    sampled, _ = sample_farthest_points(
        pts_t,
        K=n,
        random_start_point=False,
    )
    sampled_np = sampled[0].cpu().numpy().astype(np.float32)

    rng = np.random.default_rng(int(seed))
    extra_idx = rng.choice(n, size=(k - n), replace=True)
    extra_np = sampled_np[extra_idx]
    return np.concatenate([sampled_np, extra_np], axis=0).astype(np.float32)


def _load_checkpoint_flexible(
    model, checkpoint_path: Path, device: torch.device
) -> None:
    try:
        load_model_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            map_location=device,
            strict=True,
            weights_only=True,
        )
        return
    except Exception:
        pass

    raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(raw, Mapping):
        state = raw.get("model_state", raw)
    else:
        raise TypeError(
            f"Unsupported checkpoint type for {checkpoint_path}: {type(raw).__name__}"
        )

    if isinstance(state, Mapping):
        try:
            model.backbone.load_state_dict(state)
            return
        except Exception:
            # Try removing/adding DataParallel prefixes.
            if any(str(k).startswith("module.") for k in state.keys()):
                stripped = {
                    str(k).replace("module.", "", 1): v for k, v in state.items()
                }
                model.backbone.load_state_dict(stripped)
                return
            prefixed = {f"module.{k}": v for k, v in state.items()}
            model.backbone.load_state_dict(prefixed)
            return

    raise RuntimeError(f"Unable to load checkpoint weights from {checkpoint_path}")


def _build_patch_dataset(
    defected_np: np.ndarray,
    original_np: np.ndarray,
    *,
    patch_radius: float,
    points_per_patch: int,
    seed: int,
    cache_capacity: int,
    shape_name: str,
    use_pca: bool,
    patch_center: str,
    point_tuple: int,
) -> PointcloudPatchDataset:
    ds = _SingleCloudDataset(defected=defected_np, original=original_np)
    return PointcloudPatchDataset(
        dataset=ds,
        patch_radius=[float(patch_radius)],
        points_per_patch=int(points_per_patch),
        patch_features=["original"],
        seed=int(seed),
        use_pca=bool(use_pca),
        center=str(patch_center),
        point_tuple=int(point_tuple),
        sparse_patches=False,
        cache_capacity=int(cache_capacity),
        shape_names=[shape_name],
    )


def _predict_denoised_centers(
    *,
    model,
    dataloader: DataLoader,
    device: torch.device,
    use_pca: bool,
    use_point_stn: bool,
    total_patches: int,
    shape_name: str,
) -> torch.Tensor:
    out = torch.zeros(total_patches, 3, dtype=torch.float32)
    patch_offset = 0

    with torch.no_grad():
        for batchind, data in enumerate(dataloader):
            points, originals, patch_radiuses, data_trans = data
            points = points.transpose(2, 1).to(device)
            originals = originals.to(device)
            patch_radiuses = patch_radiuses.to(device)
            data_trans = data_trans.to(device)

            pred, trans, _, _ = model.backbone(points)

            o_pred = pred[:, :3]
            if trans is not None and bool(use_point_stn):
                o_pred = torch.bmm(o_pred.unsqueeze(1), trans.transpose(2, 1)).squeeze(
                    1
                )
            if bool(use_pca):
                o_pred = torch.bmm(
                    o_pred.unsqueeze(1), data_trans.transpose(2, 1)
                ).squeeze(1)

            n_points = patch_radiuses.shape[0]
            o_pred = (
                o_pred * torch.t(patch_radiuses.expand(3, n_points)).float()
                + originals.float()
            )

            batch_count = int(o_pred.shape[0])
            out[patch_offset : patch_offset + batch_count] = o_pred.detach().cpu()
            patch_offset += batch_count

            print(f"[denoise {batchind + 1}/{len(dataloader)}] shape={shape_name}")

    if patch_offset != total_patches:
        raise RuntimeError(
            f"Predicted patch count mismatch: got {patch_offset}, expected {total_patches}"
        )

    return out


def _predict_outlier_scores(
    *,
    model,
    dataloader: DataLoader,
    device: torch.device,
    total_patches: int,
    shape_name: str,
) -> torch.Tensor:
    scores = torch.zeros(total_patches, dtype=torch.float32)
    patch_offset = 0

    with torch.no_grad():
        for batchind, data in enumerate(dataloader):
            points, _, _, _ = data
            points = points.transpose(2, 1).to(device)

            pred, _, _, _ = model.backbone(points)
            outlier_score = pred[:, 0].detach().cpu().float()

            batch_count = int(outlier_score.shape[0])
            scores[patch_offset : patch_offset + batch_count] = outlier_score
            patch_offset += batch_count

            print(f"[outlier {batchind + 1}/{len(dataloader)}] shape={shape_name}")

    if patch_offset != total_patches:
        raise RuntimeError(
            f"Outlier score count mismatch: got {patch_offset}, expected {total_patches}"
        )

    return scores


def _apply_outlier_filter(
    *,
    cloud_np: np.ndarray,
    reference_np: np.ndarray,
    outlier_model,
    patch_radius: float,
    points_per_patch: int,
    seed: int,
    cache_capacity: int,
    use_pca: bool,
    patch_center: str,
    point_tuple: int,
    batch_size: int,
    workers: int,
    threshold: float,
    shape_name: str,
    stage_label: str,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    outlier_dataset = _build_patch_dataset(
        defected_np=cloud_np,
        original_np=reference_np,
        patch_radius=patch_radius,
        points_per_patch=points_per_patch,
        seed=seed,
        cache_capacity=cache_capacity,
        shape_name=shape_name,
        use_pca=use_pca,
        patch_center=patch_center,
        point_tuple=point_tuple,
    )
    outlier_loader = DataLoader(
        outlier_dataset,
        batch_size=batch_size,
        num_workers=workers,
    )

    outlier_scores = _predict_outlier_scores(
        model=outlier_model,
        dataloader=outlier_loader,
        device=next(outlier_model.parameters()).device,
        total_patches=int(outlier_dataset.shape_patch_count[0]),
        shape_name=f"{shape_name}:{stage_label}",
    )
    outlier_mask = outlier_scores > float(threshold)
    keep_mask = ~outlier_mask

    kept = int(keep_mask.sum().item())
    total = int(keep_mask.numel())
    print(
        f"[outlier:{stage_label}] removed={total - kept}/{total} threshold={float(threshold):.4f}"
    )

    if kept < max(points_per_patch, 32):
        raise RuntimeError(
            f"Too few points left after outlier filtering ({stage_label}): {kept}. "
            "Lower --outlier-threshold or disable this stage."
        )

    keep_mask_np = keep_mask.cpu().numpy()
    filtered_cloud = cloud_np[keep_mask_np]
    if reference_np.shape[0] == cloud_np.shape[0]:
        filtered_reference = reference_np[keep_mask_np]
    else:
        # Reference may be a different cloud (e.g., clean target with different
        # cardinality). In that case, keep it unchanged.
        filtered_reference = reference_np
    return filtered_cloud, filtered_reference, outlier_mask, outlier_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PointCleanNet inference on thesis dataset: optional outlier removal + denoising",
    )
    parser.add_argument(
        "--checkpoint-model",
        type=str,
        default="outputs/pointcleannet/PointCleanNet_model.pth",
        help="Path to denoising model checkpoint (*.pth).",
    )
    parser.add_argument(
        "--checkpoint-params",
        type=str,
        default="outputs/pointcleannet/PointCleanNet_params.pth",
        help="Optional params file used for default eval settings.",
    )
    parser.add_argument(
        "--run-outlier-removal",
        action="store_true",
        default=False,
        help="Run official-style outlier scoring and remove outliers before denoising.",
    )
    parser.add_argument(
        "--run-outlier-removal-after-denoise",
        action="store_true",
        default=False,
        help="Run an additional outlier-removal pass on denoised output.",
    )
    parser.add_argument(
        "--outlier-checkpoint-model",
        type=str,
        default="",
        help="Path to outlier-removal checkpoint (*.pth). Required when --run-outlier-removal is set.",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=0.9,
        help="Official-inspired threshold on outlier score (score > threshold => remove).",
    )
    parser.add_argument(
        "--dataset-variant",
        type=str,
        default="advanced",
        choices=["basic", "advanced"],
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--shape-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache-capacity", type=int, default=100)
    parser.add_argument("--seed", type=int, default=40938661)
    parser.add_argument("--n-neighbours", type=int, default=100)
    parser.add_argument(
        "--patch-radius",
        type=float,
        default=0.0,
        help="Patch radius fraction of bbox diagonal; <=0 uses params/default.",
    )
    parser.add_argument(
        "--points-per-patch",
        type=int,
        default=0,
        help="Points per patch; <=0 uses params/default.",
    )
    parser.add_argument("--dense", action="store_true", default=True)
    parser.add_argument("--no-dense", dest="dense", action="store_false")
    parser.add_argument(
        "--official-root",
        type=str,
        default=None,
        help="Unused compatibility argument (kept for CLI backward compatibility).",
    )
    args = parser.parse_args()

    cfg = bootstrap(seed=int(args.seed), data_subdir=None)
    device = cfg.device

    checkpoint_model = Path(args.checkpoint_model).expanduser().resolve()
    if not checkpoint_model.exists():
        raise FileNotFoundError(f"Missing denoise checkpoint: {checkpoint_model}")

    checkpoint_params = Path(args.checkpoint_params).expanduser().resolve()
    trainopt = None
    if checkpoint_params.exists():
        trainopt = torch.load(checkpoint_params, map_location="cpu", weights_only=False)

    points_per_patch = (
        int(args.points_per_patch)
        if int(args.points_per_patch) > 0
        else int(getattr(trainopt, "points_per_patch", 500))
    )
    patch_radius = (
        float(args.patch_radius)
        if float(args.patch_radius) > 0.0
        else _safe_first_patch_radius(getattr(trainopt, "patch_radius", [0.05]), 0.05)
    )
    model_batch_size = (
        int(args.batch_size)
        if int(args.batch_size) > 0
        else int(getattr(trainopt, "batchSize", 128))
    )

    use_pca = bool(getattr(trainopt, "use_pca", False))
    patch_center = str(getattr(trainopt, "patch_center", "point"))
    point_tuple = int(getattr(trainopt, "point_tuple", 1))
    use_point_stn = bool(getattr(trainopt, "use_point_stn", True))
    use_feat_stn = bool(getattr(trainopt, "use_feat_stn", True))
    sym_op = str(getattr(trainopt, "sym_op", "max"))

    dataset_factory = (
        create_advanced_reconstruction_dataset
        if str(args.dataset_variant).lower() == "advanced"
        else create_basic_reconstruction_dataset
    )
    base_dataset = dataset_factory(
        root=str(cfg.data_dir / "ShapeNetV2"),
        seed=int(args.seed),
        dense=bool(args.dense),
        dense_root=str(cfg.data_dir / "ShapeNetV2_dense"),
        split_into_patches=False,
        normalize=True,
    )

    sample_index = int(args.sample_index)
    if sample_index < 0 or sample_index >= len(base_dataset):
        raise IndexError(
            f"sample-index out of range: {sample_index} (dataset len={len(base_dataset)})"
        )

    sample = base_dataset[sample_index]
    defected_np, original_np = _extract_cloud_pair(sample)
    shape_name = args.shape_name or f"sample_{sample_index:07d}"

    outlier_mask = None
    filtered_input_np = defected_np
    filtered_original_np = original_np

    run_outlier_before = bool(args.run_outlier_removal)
    run_outlier_after = bool(args.run_outlier_removal_after_denoise)
    outlier_model = None

    if run_outlier_before or run_outlier_after:
        outlier_ckpt = Path(args.outlier_checkpoint_model).expanduser().resolve()
        if not args.outlier_checkpoint_model or not outlier_ckpt.exists():
            raise FileNotFoundError(
                "Outlier stage requires a valid --outlier-checkpoint-model"
            )

        outlier_model = create_model(
            "pointcleannet_outliers",
            {
                "num_points": points_per_patch,
                "num_scales": 1,
                "output_dim": 1,
                "use_point_stn": use_point_stn,
                "use_feat_stn": use_feat_stn,
                "sym_op": sym_op,
                "point_tuple": point_tuple,
            },
            device=device,
        )
        _load_checkpoint_flexible(outlier_model, outlier_ckpt, device)
        outlier_model.eval()

    if run_outlier_before:
        filtered_input_np, filtered_original_np, outlier_mask, _ = (
            _apply_outlier_filter(
                cloud_np=defected_np,
                reference_np=original_np,
                outlier_model=outlier_model,
                patch_radius=patch_radius,
                points_per_patch=points_per_patch,
                seed=int(args.seed),
                cache_capacity=int(args.cache_capacity),
                use_pca=use_pca,
                patch_center=patch_center,
                point_tuple=point_tuple,
                batch_size=model_batch_size,
                workers=int(args.workers),
                threshold=float(args.outlier_threshold),
                shape_name=shape_name,
                stage_label="before_denoise",
            )
        )

    denoise_dataset = _build_patch_dataset(
        defected_np=filtered_input_np,
        original_np=filtered_original_np,
        patch_radius=patch_radius,
        points_per_patch=points_per_patch,
        seed=int(args.seed),
        cache_capacity=int(args.cache_capacity),
        shape_name=shape_name,
        use_pca=use_pca,
        patch_center=patch_center,
        point_tuple=point_tuple,
    )
    denoise_loader = DataLoader(
        denoise_dataset,
        batch_size=model_batch_size,
        num_workers=int(args.workers),
    )

    denoise_model = create_model(
        "pointcleannet",
        {
            "num_points": points_per_patch,
            "num_scales": 1,
            "output_dim": 3,
            "use_point_stn": use_point_stn,
            "use_feat_stn": use_feat_stn,
            "sym_op": sym_op,
            "point_tuple": point_tuple,
        },
        device=device,
    )
    _load_checkpoint_flexible(denoise_model, checkpoint_model, device)
    denoise_model.eval()

    shape_properties = _predict_denoised_centers(
        model=denoise_model,
        dataloader=denoise_loader,
        device=device,
        use_pca=use_pca,
        use_point_stn=use_point_stn,
        total_patches=int(denoise_dataset.shape_patch_count[0]),
        shape_name=shape_name,
    )

    shp = denoise_dataset.shape_cache.get(0)
    pts = torch.tensor(shp.pts, dtype=torch.float32)

    n_nei = max(1, min(int(args.n_neighbours), int(pts.shape[0])))
    nearest_neighbours = torch.tensor(
        shp.kdtree.query(shp.pts, n_nei)[1], dtype=torch.long
    )
    displacement_vectors = shape_properties - pts
    mean_nei_disp = displacement_vectors[nearest_neighbours].mean(1)
    denoised_full = shape_properties - mean_nei_disp
    denoised_np = denoised_full.numpy().astype(np.float32)

    outlier_mask_after = None
    if run_outlier_after:
        if shp.clean_points is not None:
            reference_after = np.asarray(shp.clean_points, dtype=np.float32)
        else:
            reference_after = denoised_np

        denoised_np, _, outlier_mask_after, _ = _apply_outlier_filter(
            cloud_np=denoised_np,
            reference_np=reference_after,
            outlier_model=outlier_model,
            patch_radius=patch_radius,
            points_per_patch=points_per_patch,
            seed=int(args.seed),
            cache_capacity=int(args.cache_capacity),
            use_pca=use_pca,
            patch_center=patch_center,
            point_tuple=point_tuple,
            batch_size=model_batch_size,
            workers=int(args.workers),
            threshold=float(args.outlier_threshold),
            shape_name=shape_name,
            stage_label="after_denoise",
        )

    completionmodel = create_model(
        "pointr",
        {
            "trans_dim": 384,
            "knn_layer": 1,
            "num_pred": 16384,
            "num_query": 224,
        },
        device=device,
    )
    load_model_checkpoint(
        model=completionmodel,
        checkpoint_path=cfg.checkpoint_dir / "pointr" / "v1_best.pt",
        map_location=device,
        strict=True,
        weights_only=True,
    )
    completionmodel.eval()

    completion_input_np = _downsample_points_to_fixed_count(
        denoised_np,
        target_count=8192,
        seed=int(args.seed),
    )

    with torch.no_grad():
        completion_input_t = (
            torch.from_numpy(completion_input_np).float().unsqueeze(0).to(device)
        )
        _, completion_out = completionmodel(completion_input_t)

    if isinstance(completion_out, (tuple, list)):
        completed_t = completion_out[-1]
    else:
        completed_t = completion_out

    if (
        not torch.is_tensor(completed_t)
        or completed_t.ndim != 3
        or completed_t.shape[-1] != 3
    ):
        raise RuntimeError(
            "Unexpected completion output format. Expected tensor of shape (B,N,3)."
        )

    completed_np = completed_t[0].detach().cpu().numpy().astype(np.float32)

    import polyscope as ps

    ps.init()

    if args.run_outlier_removal and outlier_mask is not None:
        ps.register_point_cloud(
            "input_full_before_outlier",
            defected_np.astype(np.float32),
            point_render_mode="quad",
            radius=0.00222,
        )

    defected_vis = pts.numpy().astype(np.float32)

    ps.register_point_cloud(
        "defected_full",
        defected_vis,
        point_render_mode="quad",
        radius=0.00222,
    )

    ps.register_point_cloud(
        "original_full_subsampled",
        completion_input_np,
        point_render_mode="quad",
        radius=0.00222,
    )
    if run_outlier_after and outlier_mask_after is not None:
        ps.register_point_cloud(
            "denoised_full_before_outlier",
            denoised_full.numpy().astype(np.float32),
            point_render_mode="quad",
            radius=0.00222,
        )
    ps.register_point_cloud(
        "denoised_full",
        denoised_np,
        point_render_mode="quad",
        radius=0.00222,
    )
    ps.register_point_cloud(
        "completed_full",
        completed_np,
        point_render_mode="quad",
        radius=0.00222,
    )

    if shp.clean_points is not None:
        ps.register_point_cloud(
            "original_full",
            np.asarray(shp.clean_points, dtype=np.float32),
            point_render_mode="quad",
        )

    ps.show()


if __name__ == "__main__":
    main()
