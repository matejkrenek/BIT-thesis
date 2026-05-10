"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: infer.py
Responsibility: CLI entry point for standalone point cloud inference and optional Polyscope visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.logger import logger
from core.inference_pipeline import (
    InferenceOptions,
    run_inference,
    visualize_result_polyscope,
)


def _default_output_path(input_path: Path) -> Path:
    """Return default output .npz path for a given input file."""
    stem = input_path.stem
    return Path("outputs") / "eval" / f"{stem}_inference_out.npz"


def _default_report_path(output_path: Path) -> Path:
    """Return default JSON report path next to the output file."""
    return output_path.with_suffix(".json")


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve and normalize input/output/report paths from CLI args."""
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output_path(input_path).resolve()
    )
    report_path = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else _default_report_path(output_path).resolve()
    )
    return input_path, output_path, report_path


def _build_inference_options(
    args: argparse.Namespace,
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> InferenceOptions:
    """Build InferenceOptions from parsed CLI args and resolved paths."""
    run_denoise = bool(args.run_denoise)
    run_completion = bool(args.run_completion)
    if not run_denoise and not run_completion:
        run_denoise = True

    return InferenceOptions(
        input_path=str(input_path),
        output_path=str(output_path),
        report_json_path=str(report_path),
        input_npz_key=str(args.input_npz_key),
        output_npz_key=str(args.output_npz_key),
        seed=int(args.seed),
        workers=int(args.workers),
        batch_size=int(args.batch_size),
        cache_capacity=int(args.cache_capacity),
        n_neighbours=int(args.n_neighbours),
        patch_radius=float(args.patch_radius),
        points_per_patch=int(args.points_per_patch),
        completion_points=int(args.completion_points),
        outlier_threshold=float(args.outlier_threshold),
        point_radius=float(args.point_radius),
        run_outlier_before=bool(args.run_outlier_before),
        run_denoise=run_denoise,
        run_outlier_after=bool(args.run_outlier_after),
        run_completion=run_completion,
        completion_model=str(args.completion_model),
        denoise_checkpoint=str(args.denoise_checkpoint),
        denoise_params_checkpoint=str(args.denoise_params_checkpoint),
        outlier_checkpoint=str(args.outlier_checkpoint),
        completion_checkpoint=str(args.completion_checkpoint),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for standalone inference pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Standalone point-cloud pipeline inference: outlier removal, denoising, completion.",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input point cloud file (.npz or .ply). For colorized PLY, RGB channels are ignored and only XYZ is used.",
    )
    parser.add_argument(
        "--input-npz-key",
        type=str,
        default="points",
        help="Preferred NPZ key for input points. Falls back to first (N,3) key when missing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file path (.npz or .ply). Default: outputs/eval/<input>_inference_out.npz",
    )
    parser.add_argument(
        "--output-npz-key",
        type=str,
        default="points",
        help="NPZ key used when saving to .npz.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Path to JSON inference report. Default: <output>.json",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-capacity", type=int, default=100)
    parser.add_argument("--n-neighbours", type=int, default=100)
    parser.add_argument("--patch-radius", type=float, default=0.05)
    parser.add_argument("--points-per-patch", type=int, default=500)
    parser.add_argument("--completion-points", type=int, default=8192)
    parser.add_argument("--outlier-threshold", type=float, default=0.6)
    parser.add_argument("--point-radius", type=float, default=0.0022)

    parser.add_argument("--run-outlier-before", action="store_true", default=True)
    parser.add_argument("--run-outlier-after", action="store_true", default=True)
    parser.add_argument("--run-denoise", action="store_true", default=True)
    parser.add_argument("--run-completion", action="store_true", default=True)

    parser.add_argument(
        "--completion-model",
        type=str,
        default="adapointr",
        choices=["pcn", "pointr", "adapointr"],
        help="Completion backend model.",
    )

    parser.add_argument(
        "--denoise-checkpoint",
        type=str,
        required=True,
        help="Checkpoint path for denoising model.",
    )
    parser.add_argument(
        "--denoise-params-checkpoint",
        type=str,
        default="",
        help="Optional params checkpoint for denoising defaults.",
    )
    parser.add_argument(
        "--outlier-checkpoint",
        type=str,
        required=True,
        help="Checkpoint path for outlier model.",
    )
    parser.add_argument(
        "--completion-checkpoint",
        type=str,
        required=True,
        help="Checkpoint path for completion model.",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize all available pipeline stages with Polyscope.",
    )

    return parser


def main() -> None:
    """Parse CLI args, run inference pipeline, and print concise summary."""
    parser = build_parser()
    args = parser.parse_args()

    input_path, output_path, report_path = _resolve_paths(args)
    options = _build_inference_options(
        args,
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
    )

    result = run_inference(options)

    logger.info("[inference] input points: {}", result.input_info["points"])
    logger.info("[inference] output points: {}", result.output_info["points"])
    for stage in result.stage_timings:
        logger.info("[timing] {}: {:.4f}s", stage.name, stage.seconds)
    logger.info("[saved] output: {}", output_path)
    logger.info("[saved] report: {}", report_path)

    if bool(args.visualize):
        visualize_result_polyscope(result)


if __name__ == "__main__":
    main()
