"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: cli_parsing.py
Responsibility: Shared CLI parsing helpers for common comma-separated and view/format argument patterns.
"""

from __future__ import annotations


def parse_csv(value: str) -> list[str]:
    """Parse comma-separated values into a stripped list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_csv(value: str | None) -> list[str] | None:
    """Parse optional comma-separated values and return None when empty."""
    if not value:
        return None
    parsed = parse_csv(value)
    return parsed or None


def parse_views(value: str) -> list[tuple[float, float]]:
    """Parse semicolon-separated view pairs in 'elev,azim' format."""
    views: list[tuple[float, float]] = []
    for pair in value.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        elev_str, azim_str = pair.split(",")
        views.append((float(elev_str), float(azim_str)))

    if not views:
        raise ValueError("At least one view must be provided")
    return views


def parse_xyz_degrees(value: str) -> tuple[float, float, float]:
    """Parse XYZ rotation in degrees from 'x,y,z'."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(
            "Expected three comma-separated values for XYZ rotation, e.g. 0,0,90"
        )
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_indices(value: str, *, arg_name: str = "--sample-indices") -> list[int]:
    """Parse comma-separated integer indices."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{arg_name} was provided but no indices were parsed")
    return [int(part) for part in parts]


def parse_labels(value: str, *, arg_name: str = "--cloud-labels") -> list[str]:
    """Parse comma-separated cloud labels."""
    labels = [part.strip() for part in value.split(",") if part.strip()]
    if not labels:
        raise ValueError(f"{arg_name} was provided but no labels were parsed")
    return labels


def parse_output_formats(value: str) -> list[str]:
    """Parse and validate output format list for gallery export."""
    supported = {"png", "svg", "pdf"}
    raw = [
        item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()
    ]
    if not raw:
        raise ValueError("--export-formats must contain at least one format")

    invalid = [fmt for fmt in raw if fmt not in supported]
    if invalid:
        raise ValueError(f"Unsupported format(s) {invalid}. Choose from: png, svg, pdf")

    unique: list[str] = []
    seen: set[str] = set()
    for fmt in raw:
        if fmt not in seen:
            seen.add(fmt)
            unique.append(fmt)
    return unique


__all__ = [
    "parse_csv",
    "parse_optional_csv",
    "parse_views",
    "parse_xyz_degrees",
    "parse_indices",
    "parse_labels",
    "parse_output_formats",
]
