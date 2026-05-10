from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.bootstrap import bootstrap
from core.model_defaults import get_default_model_params
from core.models import create_model, load_model_checkpoint
from dataset.wrapper import PointcloudPatchDataset


@dataclass
class InferenceOptions:
    input_path: str
    output_path: str
    report_json_path: str
    input_npz_key: str = "points"
    output_npz_key: str = "points"
    seed: int = 40938661
    workers: int = 1
    batch_size: int = 128
    cache_capacity: int = 100
    n_neighbours: int = 100
    patch_radius: float = 0.05
    points_per_patch: int = 500
    completion_points: int = 8192
    outlier_threshold: float = 0.4
    point_radius: float = 0.0022
    run_outlier_before: bool = False
    run_denoise: bool = True
    run_outlier_after: bool = False
    run_completion: bool = False
    completion_model: str = "adapointr"
    denoise_checkpoint: str = ""
    denoise_params_checkpoint: str = ""
    outlier_checkpoint: str = ""
    completion_checkpoint: str = ""


@dataclass
class StageTiming:
    name: str
    seconds: float


@dataclass
class InferenceArtifacts:
    input_cloud: np.ndarray
    outlier_filtered_before: np.ndarray | None
    denoised: np.ndarray | None
    outlier_filtered_after: np.ndarray | None
    completion_input: np.ndarray | None
    completion_centers: np.ndarray | None
    completed: np.ndarray | None
    output_cloud: np.ndarray


@dataclass
class InferenceResult:
    options: InferenceOptions
    stage_timings: list[StageTiming]
    input_info: dict[str, Any]
    output_info: dict[str, Any]
    artifacts: InferenceArtifacts

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "options": asdict(self.options),
            "stage_timings": [asdict(stage) for stage in self.stage_timings],
            "input_info": self.input_info,
            "output_info": self.output_info,
            "summary": {
                "total_seconds": float(sum(s.seconds for s in self.stage_timings)),
                "stage_count": int(len(self.stage_timings)),
            },
        }


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


def _safe_first_patch_radius(value, fallback: float) -> float:
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return float(value[0])
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _timed(
    stage: str,
    timings: list[StageTiming],
    fn,
    *args,
    progress: dict[str, int] | None = None,
    **kwargs,
):
    if progress is not None:
        start_idx = int(progress.get("done", 0)) + 1
        total = int(progress.get("total", 0))
        print(f"[progress {start_idx}/{total}] START {stage}")

    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    timings.append(StageTiming(name=stage, seconds=elapsed))

    if progress is not None:
        progress["done"] = int(progress.get("done", 0)) + 1
        done = int(progress.get("done", 0))
        total = int(progress.get("total", 0))
        print(f"[progress {done}/{total}] DONE {stage} ({elapsed:.3f}s)")

    return out


def _load_point_cloud(path: Path, npz_key: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        npz = np.load(str(path))
        if npz_key in npz:
            arr = np.asarray(npz[npz_key], dtype=np.float32)
        else:
            arr = None
            for key in npz.files:
                candidate = np.asarray(npz[key])
                if candidate.ndim == 2 and candidate.shape[1] == 3:
                    arr = np.asarray(candidate, dtype=np.float32)
                    break
            if arr is None:
                raise ValueError(
                    f"NPZ does not contain an (N,3) array. Keys: {list(npz.files)}"
                )
        return _ensure_points(arr)

    if suffix == ".ply":
        try:
            import trimesh

            mesh = trimesh.load(str(path), process=False)
            if isinstance(mesh, trimesh.Scene):
                if not mesh.geometry:
                    raise ValueError(f"PLY scene contains no geometry: {path}")
                mesh = next(iter(mesh.geometry.values()))

            if hasattr(mesh, "vertices"):
                points = np.asarray(mesh.vertices, dtype=np.float32)
            elif hasattr(mesh, "points"):
                points = np.asarray(mesh.points, dtype=np.float32)
            else:
                points = np.asarray(mesh, dtype=np.float32)
        except Exception:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points, dtype=np.float32)
        return _ensure_points(points)

    raise ValueError(f"Unsupported input format: {path.suffix}. Use .npz or .ply")


def _save_point_cloud(path: Path, points: np.ndarray, npz_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        np.savez_compressed(
            str(path), **{npz_key: np.asarray(points, dtype=np.float32)}
        )
        return

    if suffix == ".ply":
        pts = np.asarray(points, dtype=np.float32)
        try:
            import trimesh

            trimesh.PointCloud(pts).export(str(path))
        except Exception:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            o3d.io.write_point_cloud(str(path), pcd)
        return

    raise ValueError(f"Unsupported output format: {path.suffix}. Use .npz or .ply")


def _ensure_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Point cloud must have shape (N,3), got {pts.shape}")
    # Keep XYZ only; extra channels (e.g., RGB from photogrammetric PLY) are ignored.
    return np.ascontiguousarray(pts[:, :3], dtype=np.float32)


def _downsample_points_max(
    points: np.ndarray, *, max_points: int, seed: int
) -> np.ndarray:
    pts = _ensure_points(points)
    max_pts = int(max_points)
    if max_pts <= 0 or int(pts.shape[0]) <= max_pts:
        return pts

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(int(pts.shape[0]), size=max_pts, replace=False)
    return np.ascontiguousarray(pts[idx], dtype=np.float32)


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
    progress_desc: str,
) -> torch.Tensor:
    out = torch.zeros(total_patches, 3, dtype=torch.float32)
    patch_offset = 0

    with torch.no_grad():
        for data in tqdm(
            dataloader,
            desc=progress_desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
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
    progress_desc: str,
) -> torch.Tensor:
    scores = torch.zeros(total_patches, dtype=torch.float32)
    patch_offset = 0

    with torch.no_grad():
        for data in tqdm(
            dataloader,
            desc=progress_desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ):
            points, _, _, _ = data
            points = points.transpose(2, 1).to(device)

            pred, _, _, _ = model.backbone(points)
            outlier_score = pred[:, 0].detach().cpu().float()

            batch_count = int(outlier_score.shape[0])
            scores[patch_offset : patch_offset + batch_count] = outlier_score
            patch_offset += batch_count

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
    stage_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    outlier_dataset = _build_patch_dataset(
        defected_np=cloud_np,
        original_np=reference_np,
        patch_radius=patch_radius,
        points_per_patch=points_per_patch,
        seed=seed,
        cache_capacity=cache_capacity,
        shape_name="inference",
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
        progress_desc=f"outlier:{stage_label}",
    )

    keep_mask = ~(outlier_scores > float(threshold))
    kept = int(keep_mask.sum().item())
    if kept < max(points_per_patch, 32):
        raise RuntimeError(
            f"Too few points left after outlier filtering: {kept}. "
            "Lower --outlier-threshold or disable outlier stage."
        )

    keep_mask_np = keep_mask.cpu().numpy()
    filtered_cloud = cloud_np[keep_mask_np]
    if reference_np.shape[0] == cloud_np.shape[0]:
        filtered_reference = reference_np[keep_mask_np]
    else:
        filtered_reference = reference_np

    return filtered_cloud, filtered_reference


def _run_completion_with_fps_input(
    *,
    completion_model,
    points_np: np.ndarray,
    device: torch.device,
    input_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = _ensure_points(points_np)

    target_input = max(1, int(input_points))
    in_t = torch.from_numpy(pts).unsqueeze(0).to(device=device, dtype=torch.float32)

    if in_t.shape[1] >= target_input:
        in_t, _ = sample_farthest_points(in_t, K=target_input)
    else:
        rng = np.random.default_rng(int(seed))
        extra_idx = rng.choice(
            int(in_t.shape[1]), size=target_input - int(in_t.shape[1]), replace=True
        )
        extra_t = in_t[
            :, torch.as_tensor(extra_idx, device=device, dtype=torch.long), :
        ]
        in_t = torch.cat([in_t, extra_t], dim=1)

    with torch.no_grad():
        raw_out = completion_model(in_t)

    if isinstance(raw_out, tuple) and len(raw_out) == 2:
        completion_out = raw_out[1]
    else:
        completion_out = raw_out

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
            "Unexpected completion output format. Expected tensor (B,N,3)."
        )

    input_np = in_t[0].detach().cpu().numpy().astype(np.float32)
    completed_np = completed_t[0].detach().cpu().numpy().astype(np.float32)
    centers_np = np.asarray(input_np.mean(axis=0, keepdims=True), dtype=np.float32)
    return completed_np, input_np, centers_np


def _make_completion_config(model_name: str) -> dict[str, Any]:
    name = model_name.strip().lower()
    if name not in ("pcn", "pointr", "adapointr"):
        raise ValueError(
            "Unsupported completion model. Use one of: "
            f"{', '.join(("pcn", "pointr", "adapointr"))}"
        )

    # Validate model availability via centralized constructor defaults.
    get_default_model_params(name)
    return {"model_name": name}


def run_inference(options: InferenceOptions) -> InferenceResult:
    input_path = Path(options.input_path).expanduser().resolve()
    output_path = Path(options.output_path).expanduser().resolve()
    report_path = Path(options.report_json_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    planned_stages: list[str] = ["load_input"]
    if options.run_outlier_before or options.run_outlier_after:
        planned_stages.append("load_outlier_model")
    if options.run_outlier_before:
        planned_stages.append("outlier_before")
    if options.run_denoise:
        planned_stages.append("denoise")
    if options.run_outlier_after:
        planned_stages.append("outlier_after")
    if options.run_completion:
        planned_stages.extend(["load_completion_model", "completion"])
    planned_stages.append("save_output")

    progress = {"done": 0, "total": len(planned_stages)}
    print(
        f"[progress] Pipeline stages ({len(planned_stages)}): "
        + " -> ".join(planned_stages)
    )

    cfg = bootstrap(seed=int(options.seed), data_subdir=None)
    device = cfg.device

    timings: list[StageTiming] = []

    input_cloud = _timed(
        "load_input",
        timings,
        _load_point_cloud,
        input_path,
        str(options.input_npz_key),
        progress=progress,
    )
    input_before_downsample = int(input_cloud.shape[0])
    input_cloud = _downsample_points_max(
        input_cloud,
        max_points=200000,
        seed=int(options.seed),
    )
    if int(input_cloud.shape[0]) != input_before_downsample:
        print(
            "[input] downsampled point cloud: "
            f"{input_before_downsample} -> {int(input_cloud.shape[0])} points"
        )
    print(f"[input] loaded point cloud: {int(input_cloud.shape[0])} points")

    denoise_model_name = "pointcleannet"
    denoise_default_params = get_default_model_params(denoise_model_name)
    params_path = Path(options.denoise_params_checkpoint).expanduser().resolve()
    trainopt = None
    if params_path.exists():
        trainopt = torch.load(params_path, map_location="cpu", weights_only=False)

    points_per_patch = int(options.points_per_patch)
    if points_per_patch <= 0:
        points_per_patch = int(
            getattr(trainopt, "points_per_patch", denoise_default_params["num_points"])
        )

    patch_radius = float(options.patch_radius)
    if patch_radius <= 0.0:
        patch_radius = _safe_first_patch_radius(
            getattr(trainopt, "patch_radius", [0.05]), 0.05
        )

    batch_size = max(1, int(options.batch_size))

    use_pca = bool(getattr(trainopt, "use_pca", False))
    patch_center = str(getattr(trainopt, "patch_center", "point"))
    point_tuple = int(getattr(trainopt, "point_tuple", 1))
    use_point_stn = bool(getattr(trainopt, "use_point_stn", True))
    use_feat_stn = bool(getattr(trainopt, "use_feat_stn", True))
    sym_op = str(getattr(trainopt, "sym_op", "max"))

    if options.run_denoise:
        if not str(options.denoise_checkpoint).strip():
            raise ValueError("Denoise stage requires --denoise-checkpoint")
        denoise_ckpt = Path(options.denoise_checkpoint).expanduser().resolve()
        if not denoise_ckpt.exists():
            raise FileNotFoundError(f"Missing denoise checkpoint: {denoise_ckpt}")

        denoise_model_params = get_default_model_params(denoise_model_name)
        denoise_model_params.update(
            {
                "num_points": int(points_per_patch),
                "use_point_stn": use_point_stn,
                "use_feat_stn": use_feat_stn,
                "sym_op": sym_op,
                "point_tuple": point_tuple,
            }
        )
    else:
        denoise_ckpt = None
        denoise_model_params = {}

    current_cloud = np.asarray(input_cloud, dtype=np.float32)
    current_reference = np.asarray(input_cloud, dtype=np.float32)

    outlier_before_cloud = None
    outlier_after_cloud = None
    denoised_cloud = None
    completed_cloud = None
    completion_input = None
    completion_centers = None

    outlier_model = None
    if options.run_outlier_before or options.run_outlier_after:
        if not str(options.outlier_checkpoint).strip():
            raise ValueError("Outlier stage requires --outlier-checkpoint")
        outlier_ckpt = Path(options.outlier_checkpoint).expanduser().resolve()
        if not outlier_ckpt.exists():
            raise FileNotFoundError(
                "Outlier stage requires a valid outlier checkpoint."
            )

        outlier_model_name = "pointcleannet_outliers"
        outlier_params = get_default_model_params(outlier_model_name)
        outlier_params.update(
            {
                "num_points": int(points_per_patch),
                "use_point_stn": use_point_stn,
                "use_feat_stn": use_feat_stn,
                "sym_op": sym_op,
                "point_tuple": point_tuple,
            }
        )

        def _build_outlier_model():
            model = create_model(
                outlier_model_name,
                outlier_params,
                device=device,
            )
            _load_checkpoint_flexible(model, outlier_ckpt, device)
            model.eval()
            return model

        outlier_model = _timed(
            "load_outlier_model",
            timings,
            _build_outlier_model,
            progress=progress,
        )

    if options.run_outlier_before:
        current_cloud, current_reference = _timed(
            "outlier_before",
            timings,
            _apply_outlier_filter,
            cloud_np=current_cloud,
            reference_np=current_reference,
            outlier_model=outlier_model,
            patch_radius=patch_radius,
            points_per_patch=points_per_patch,
            seed=int(options.seed),
            cache_capacity=int(options.cache_capacity),
            use_pca=use_pca,
            patch_center=patch_center,
            point_tuple=point_tuple,
            batch_size=batch_size,
            workers=int(options.workers),
            threshold=float(options.outlier_threshold),
            stage_label="before",
            progress=progress,
        )
        outlier_before_cloud = current_cloud.copy()

    if options.run_denoise:

        def _run_denoise_stage() -> np.ndarray:
            denoise_dataset = _build_patch_dataset(
                defected_np=current_cloud,
                original_np=current_reference,
                patch_radius=patch_radius,
                points_per_patch=points_per_patch,
                seed=int(options.seed),
                cache_capacity=int(options.cache_capacity),
                shape_name="inference",
                use_pca=use_pca,
                patch_center=patch_center,
                point_tuple=point_tuple,
            )
            denoise_loader = DataLoader(
                denoise_dataset,
                batch_size=batch_size,
                num_workers=int(options.workers),
            )

            denoise_model = create_model(
                denoise_model_name,
                denoise_model_params,
                device=device,
            )
            _load_checkpoint_flexible(denoise_model, denoise_ckpt, device)
            denoise_model.eval()

            shape_properties = _predict_denoised_centers(
                model=denoise_model,
                dataloader=denoise_loader,
                device=device,
                use_pca=use_pca,
                use_point_stn=use_point_stn,
                total_patches=int(denoise_dataset.shape_patch_count[0]),
                progress_desc="denoise",
            )

            shp = denoise_dataset.shape_cache.get(0)
            pts = torch.tensor(shp.pts, dtype=torch.float32)
            n_nei = max(1, min(int(options.n_neighbours), int(pts.shape[0])))
            nearest_neighbours = torch.tensor(
                shp.kdtree.query(shp.pts, n_nei)[1], dtype=torch.long
            )
            displacement_vectors = shape_properties - pts
            mean_nei_disp = displacement_vectors[nearest_neighbours].mean(1)
            denoised_full = shape_properties - mean_nei_disp
            return denoised_full.numpy().astype(np.float32)

        current_cloud = _timed(
            "denoise",
            timings,
            _run_denoise_stage,
            progress=progress,
        )
        denoised_cloud = current_cloud.copy()

    if options.run_outlier_after:
        current_cloud, current_reference = _timed(
            "outlier_after",
            timings,
            _apply_outlier_filter,
            cloud_np=current_cloud,
            reference_np=current_reference,
            outlier_model=outlier_model,
            patch_radius=patch_radius,
            points_per_patch=points_per_patch,
            seed=int(options.seed),
            cache_capacity=int(options.cache_capacity),
            use_pca=use_pca,
            patch_center=patch_center,
            point_tuple=point_tuple,
            batch_size=batch_size,
            workers=int(options.workers),
            threshold=float(options.outlier_threshold),
            stage_label="after",
            progress=progress,
        )
        outlier_after_cloud = current_cloud.copy()

    if options.run_completion:
        completion_cfg = _make_completion_config(options.completion_model)
        if not str(options.completion_checkpoint).strip():
            raise ValueError("Completion stage requires --completion-checkpoint")
        completion_ckpt = Path(options.completion_checkpoint).expanduser().resolve()
        if not completion_ckpt.exists():
            raise FileNotFoundError(f"Missing completion checkpoint: {completion_ckpt}")

        for _ in tqdm(
            range(1),
            desc="completion:prepare",
            unit="step",
            leave=False,
            dynamic_ncols=True,
        ):
            pass

        def _build_completion_model():
            model = create_model(
                completion_cfg["model_name"],
                get_default_model_params(completion_cfg["model_name"]),
                device=device,
            )
            load_model_checkpoint(
                model=model,
                checkpoint_path=completion_ckpt,
                map_location=device,
                strict=False,
                weights_only=True,
            )
            model.eval()
            return model

        completion_model = _timed(
            "load_completion_model",
            timings,
            _build_completion_model,
            progress=progress,
        )

        completed_cloud, completion_input, completion_centers = _timed(
            "completion",
            timings,
            _run_completion_with_fps_input,
            completion_model=completion_model,
            points_np=current_cloud,
            device=device,
            input_points=max(1, int(options.completion_points)),
            seed=int(options.seed),
            progress=progress,
        )
        current_cloud = completed_cloud.copy()

    output_cloud = _ensure_points(current_cloud)
    _timed(
        "save_output",
        timings,
        _save_point_cloud,
        output_path,
        output_cloud,
        str(options.output_npz_key),
        progress=progress,
    )

    result = InferenceResult(
        options=options,
        stage_timings=timings,
        input_info={
            "path": str(input_path),
            "points": int(input_cloud.shape[0]),
            "bbox_min": np.min(input_cloud, axis=0).tolist(),
            "bbox_max": np.max(input_cloud, axis=0).tolist(),
        },
        output_info={
            "path": str(output_path),
            "points": int(output_cloud.shape[0]),
            "bbox_min": np.min(output_cloud, axis=0).tolist(),
            "bbox_max": np.max(output_cloud, axis=0).tolist(),
            "format": output_path.suffix.lower(),
        },
        artifacts=InferenceArtifacts(
            input_cloud=input_cloud,
            outlier_filtered_before=outlier_before_cloud,
            denoised=denoised_cloud,
            outlier_filtered_after=outlier_after_cloud,
            completion_input=completion_input,
            completion_centers=completion_centers,
            completed=completed_cloud,
            output_cloud=output_cloud,
        ),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)

    return result


def visualize_result_polyscope(
    result: InferenceResult, *, point_radius: float | None = None
) -> None:
    import polyscope as ps

    radius = (
        float(point_radius)
        if point_radius is not None
        else float(result.options.point_radius)
    )

    ps.init()
    ps.register_point_cloud(
        "input_full",
        np.asarray(result.artifacts.input_cloud, dtype=np.float32),
        point_render_mode="quad",
        radius=radius,
    )

    if result.artifacts.outlier_filtered_before is not None:
        ps.register_point_cloud(
            "outlier_filtered_before",
            np.asarray(result.artifacts.outlier_filtered_before, dtype=np.float32),
            point_render_mode="quad",
            radius=radius,
        )

    if result.artifacts.denoised is not None:
        ps.register_point_cloud(
            "denoised",
            np.asarray(result.artifacts.denoised, dtype=np.float32),
            point_render_mode="quad",
            radius=radius,
        )

    if result.artifacts.outlier_filtered_after is not None:
        ps.register_point_cloud(
            "outlier_filtered_after",
            np.asarray(result.artifacts.outlier_filtered_after, dtype=np.float32),
            point_render_mode="quad",
            radius=radius,
        )

    if result.artifacts.completion_input is not None:
        ps.register_point_cloud(
            "completion_input",
            np.asarray(result.artifacts.completion_input, dtype=np.float32),
            point_render_mode="quad",
            radius=radius,
        )

    if result.artifacts.completion_centers is not None:
        ps.register_point_cloud(
            "completion_centers",
            np.asarray(result.artifacts.completion_centers, dtype=np.float32),
            point_render_mode="quad",
            radius=radius * 1.3,
        )

    if result.artifacts.completed is not None:
        ps.register_point_cloud(
            "completed",
            np.asarray(result.artifacts.completed, dtype=np.float32),
            point_render_mode="quad",
            radius=radius,
        )

    ps.register_point_cloud(
        "output_full",
        np.asarray(result.artifacts.output_cloud, dtype=np.float32),
        point_render_mode="quad",
        radius=radius,
    )

    ps.show()
