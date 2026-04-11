from __future__ import annotations

import argparse
import time
from pathlib import Path
from tqdm import tqdm

from core import (
    bootstrap,
    create_basic_reconstruction_dataset,
)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Build basic reconstruction dataset and precompute defect cache to NPZ"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--defect-augmentation-count", type=int, default=5)
    parser.add_argument("--local-dropout-regions", type=int, default=5)
    parser.add_argument("--dense", action="store_true", default=True)
    parser.add_argument("--no-dense", dest="dense", action="store_false")
    parser.add_argument("--dense-num-points", type=int, default=100_000)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for defected sample NPZ cache (default: outputs/dataset/defect_cache_basic)",
    )
    parser.add_argument("--cache-read", action="store_true", default=True)
    parser.add_argument("--no-cache-read", dest="cache_read", action="store_false")
    parser.add_argument("--cache-write", action="store_true", default=True)
    parser.add_argument("--no-cache-write", dest="cache_write", action="store_false")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="TQDM miniters hint for refresh cadence",
    )
    args = parser.parse_args()

    cfg = bootstrap()
    seed = cfg.seed if args.seed is None else int(args.seed)

    if args.cache_dir is None:
        cache_dir = cfg.output_dir / "dataset" / "defect_cache_basic"
    else:
        cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = create_basic_reconstruction_dataset(
        root=cfg.data_dir,
        seed=seed,
        defect_augmentation_count=int(args.defect_augmentation_count),
        local_dropout_regions=int(args.local_dropout_regions),
        defect_cache_npz_dir=str(cache_dir),
        defect_cache_read=bool(args.cache_read),
        defect_cache_write=bool(args.cache_write),
    )

    total = len(dataset)
    if total == 0:
        print("[INFO] Dataset is empty, nothing to cache.")
        return

    print(f"[INFO] Basic dataset size: {total}")
    print(f"[INFO] Cache dir: {cache_dir}")
    print(
        f"[INFO] Cache mode: read={bool(args.cache_read)} write={bool(args.cache_write)}"
    )

    t0 = time.time()
    progress_every = max(1, int(args.progress_every))

    for idx in tqdm(
        range(total),
        desc="Caching defected samples",
        unit="sample",
        miniters=progress_every,
    ):
        _ = dataset[idx]

    total_elapsed = time.time() - t0
    print(
        "[DONE] "
        f"Cached traversal finished: {total} samples in {_format_seconds(total_elapsed)} "
        f"({(total / total_elapsed) if total_elapsed > 0 else 0.0:.2f} samples/s)"
    )


if __name__ == "__main__":
    main()
