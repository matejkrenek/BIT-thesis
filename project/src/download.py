"""
Author: Matej Krenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: download.py
Responsibility: Download files from Hugging Face Hub into a target directory.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
from pathlib import Path


def _parse_csv_patterns(value: str | None) -> list[str] | None:
    if not value:
        return None
    patterns = [item.strip() for item in value.split(",")]
    patterns = [item for item in patterns if item]
    return patterns or None


def _matches_any(path: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _split_bucket_source(source: str) -> tuple[str, str]:
    prefix = "hf://buckets/"
    if not source.startswith(prefix):
        raise ValueError(
            "Source must start with 'hf://buckets/'. "
            "Example: hf://buckets/username/my-bucket/path"
        )

    without_scheme = source[len(prefix) :].strip("/")
    parts = without_scheme.split("/", 2)
    if len(parts) < 2:
        raise ValueError(
            "Source must include bucket owner and name. "
            "Example: hf://buckets/username/my-bucket"
        )

    bucket_id = f"{parts[0]}/{parts[1]}"
    bucket_path = parts[2] if len(parts) > 2 else ""
    return bucket_id, bucket_path


def _to_repo_relative(remote_path: str, bucket_id: str) -> str:
    # Hugging Face file listings can return different path prefixes depending on API path form.
    for candidate in (
        f"hf://buckets/{bucket_id}/",
        f"buckets/{bucket_id}/",
        f"/{bucket_id}/",
    ):
        if remote_path.startswith(candidate):
            return remote_path[len(candidate) :].lstrip("/")

    if remote_path.startswith(f"hf://buckets/{bucket_id}"):
        return remote_path[len(f"hf://buckets/{bucket_id}") :].lstrip("/")

    return remote_path.lstrip("/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download files from Hugging Face bucket storage into the selected output "
            "directory."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        default="hf://buckets/matej-krenek/BIT-thesis",
        help="Source URI. Example: hf://buckets/matej-krenek/BIT-thesis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Destination directory where repository files will be downloaded.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional HF token. If omitted, HF_TOKEN environment variable is used.",
    )
    parser.add_argument(
        "--allow-patterns",
        type=str,
        default=None,
        help="Comma-separated file patterns to include (e.g. '*.pth,checkpoints/*').",
    )
    parser.add_argument(
        "--ignore-patterns",
        type=str,
        default=None,
        help="Comma-separated file patterns to exclude.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download files even if they already exist in output directory.",
    )
    return parser


def _download_via_cli(source: str, output_dir: Path, token: str | None) -> int:
    cmd = ["hf", "buckets", "cp", source, str(output_dir)]
    if token:
        cmd.extend(["--token", token])

    print("[download.py] Falling back to Hugging Face CLI: hf buckets cp")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "CLI fallback failed. Ensure 'hf' is installed and authenticated."
        )
    print("[download.py] Download completed successfully (CLI fallback).")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.getenv("HF_TOKEN")
    allow_patterns = _parse_csv_patterns(args.allow_patterns)
    ignore_patterns = _parse_csv_patterns(args.ignore_patterns)

    if token is None:
        print("[download.py] HF token not provided; proceeding without authentication.")

    print(f"[download.py] Downloading '{args.source}' to '{output_dir}'...")

    try:
        # Prefer Python API for fine-grained filtering and per-file target paths.
        from huggingface_hub import HfFileSystem, download_bucket_files
    except ImportError:
        # Keep CLI fallback so script still works when huggingface_hub extras are missing.
        return _download_via_cli(args.source, output_dir, token)

    bucket_id, bucket_path = _split_bucket_source(args.source)
    bucket_prefix = bucket_path.strip("/")
    hf_source = f"hf://buckets/{bucket_id}"
    if bucket_prefix:
        hf_source = f"{hf_source}/{bucket_prefix}"

    fs = HfFileSystem(token=token)
    if fs.isfile(hf_source):
        # Single-file source: keep relative output rooted at the file parent folder.
        repo_paths = [_to_repo_relative(hf_source, bucket_id)]
        base_prefix = str(Path(repo_paths[0]).parent).replace("\\", "/")
    else:
        if not fs.exists(hf_source):
            raise FileNotFoundError(f"Source path does not exist: {args.source}")
        # Directory source: enumerate recursively and keep only regular files.
        found = fs.find(hf_source, maxdepth=None)
        repo_paths = [
            _to_repo_relative(remote_path, bucket_id)
            for remote_path in found
            if fs.isfile(remote_path)
        ]
        base_prefix = bucket_prefix

    if not repo_paths:
        raise RuntimeError(f"No files found at source: {args.source}")

    files_to_download: list[tuple[str, str]] = []
    skipped = 0

    for repo_path in repo_paths:
        # Preserve source sub-tree layout under output_dir.
        rel_path = os.path.relpath(repo_path, base_prefix or ".").replace("\\", "/")

        if allow_patterns and not _matches_any(rel_path, allow_patterns):
            continue
        if _matches_any(rel_path, ignore_patterns):
            continue

        local_file = output_dir / rel_path
        local_file.parent.mkdir(parents=True, exist_ok=True)

        # Default behavior is incremental download; force flag allows overwrite.
        if local_file.exists() and not args.force_download:
            skipped += 1
            continue

        files_to_download.append((repo_path, str(local_file)))

    if not files_to_download:
        print(
            "[download.py] Nothing to download after filtering/skipping. "
            f"Skipped existing: {skipped}."
        )
        return 0

    download_bucket_files(bucket_id, files=files_to_download, token=token)

    print(
        f"[download.py] Download completed successfully. Downloaded: {len(files_to_download)}, "
        f"skipped existing: {skipped}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
