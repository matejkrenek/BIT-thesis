from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from tqdm import tqdm

from core import bootstrap, create_basic_reconstruction_dataset


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _build_dataset(
    *,
    root: Path,
    seed: int,
    defect_augmentation_count: int,
    local_dropout_regions: int,
    cache_dir: Path,
    cache_read: bool,
    cache_write: bool,
):
    return create_basic_reconstruction_dataset(
        root=root,
        seed=seed,
        defect_augmentation_count=defect_augmentation_count,
        local_dropout_regions=local_dropout_regions,
        defect_cache_npz_dir=str(cache_dir),
        defect_cache_read=cache_read,
        defect_cache_write=cache_write,
    )


def _walk_dataset(
    dataset, *, desc: str, progress_every: int, show_progress: bool
) -> float:
    total = len(dataset)
    t0 = time.time()
    iterator = range(total)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc=desc,
            unit="sample",
            miniters=max(1, int(progress_every)),
        )

    for idx in iterator:
        _ = dataset[idx]

    return time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dataset traversal: on-the-fly generation vs disk-cached NPZ loading"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--defect-augmentation-count", type=int, default=5)
    parser.add_argument("--local-dropout-regions", type=int, default=5)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for NPZ cache (default: outputs/dataset/defect_cache_basic_benchmark)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete cache directory before benchmark",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--show-progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    args = parser.parse_args()

    cfg = bootstrap()

    if args.cache_dir is None:
        cache_dir = cfg.output_dir / "dataset" / "defect_cache_basic_benchmark"
    else:
        cache_dir = Path(args.cache_dir)

    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Data root: {cfg.data_dir}")
    print(f"[INFO] Cache dir: {cache_dir}")
    print(
        f"[INFO] Params: seed={args.seed}, defect_augmentation_count={args.defect_augmentation_count}, "
        f"local_dropout_regions={args.local_dropout_regions}"
    )

    ds_onthefly = _build_dataset(
        root=cfg.data_dir,
        seed=int(args.seed),
        defect_augmentation_count=int(args.defect_augmentation_count),
        local_dropout_regions=int(args.local_dropout_regions),
        cache_dir=cache_dir,
        cache_read=False,
        cache_write=False,
    )

    total = len(ds_onthefly)
    if total == 0:
        print("[INFO] Dataset is empty, nothing to benchmark.")
        return

    print(f"[INFO] Dataset size: {total}")

    # t_onthefly = _walk_dataset(
    #     ds_onthefly,
    #     desc="On-the-fly",
    #     progress_every=int(args.progress_every),
    #     show_progress=bool(args.show_progress),
    # )

    ds_cached = _build_dataset(
        root=cfg.data_dir,
        seed=int(args.seed),
        defect_augmentation_count=int(args.defect_augmentation_count),
        local_dropout_regions=int(args.local_dropout_regions),
        cache_dir=cache_dir,
        cache_read=False,
        cache_write=True,
    )
    t_cached = _walk_dataset(
        ds_cached,
        desc="Caching to disk",
        progress_every=int(args.progress_every),
        show_progress=bool(args.show_progress),
    )

    ds_from_disk = _build_dataset(
        root=cfg.data_dir,
        seed=int(args.seed),
        defect_augmentation_count=int(args.defect_augmentation_count),
        local_dropout_regions=int(args.local_dropout_regions),
        cache_dir=cache_dir,
        cache_read=True,
        cache_write=False,
    )
    t_disk = _walk_dataset(
        ds_from_disk,
        desc="From disk cache",
        progress_every=int(args.progress_every),
        show_progress=bool(args.show_progress),
    )

    # speedup = (t_onthefly / t_disk) if t_disk > 0 else 0.0

    print("\n[RESULTS]")
    # print(f"- On-the-fly traversal: {_format_seconds(t_onthefly)} ({t_onthefly:.3f}s)")
    print(f"- Caching to disk: {_format_seconds(t_cached)} ({t_cached:.3f}s)")
    print(f"- Disk-cache traversal: {_format_seconds(t_disk)} ({t_disk:.3f}s)")
    # print(f"- Speedup (on-the-fly / disk): {speedup:.2f}x")


if __name__ == "__main__":
    main()
