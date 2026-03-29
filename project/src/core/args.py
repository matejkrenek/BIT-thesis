from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .bootstrap import BootstrapConfig, bootstrap


@dataclass(frozen=True)
class ArgSpec:
    """Declarative schema entry for one CLI argument."""

    flags: tuple[str, ...]
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.flags:
            raise ValueError("ArgSpec.flags must contain at least one flag")


def build_parser(
    schema: Sequence[ArgSpec],
    *,
    description: str | None = None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Build an ArgumentParser from a simple argument schema."""
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    for arg in schema:
        parser.add_argument(*arg.flags, **dict(arg.kwargs))
    return parser


def parse_args(
    schema: Sequence[ArgSpec],
    *,
    argv: Iterable[str] | None = None,
    description: str | None = None,
    epilog: str | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments using a declarative schema."""
    parser = build_parser(schema, description=description, epilog=epilog)
    return parser.parse_args(list(argv) if argv is not None else None)


def bootstrap_from_args(
    args: argparse.Namespace,
    *,
    seed_attr: str = "seed",
    default_seed: int = 42,
    env_file: str | None = None,
    data_subdir: str = "ShapeNetV2",
    checkpoint_subdir: str | None = None,
) -> BootstrapConfig:
    """Create BootstrapConfig from parsed CLI args.

    If ``seed_attr`` is not present in args, ``default_seed`` is used.
    """
    seed = int(getattr(args, seed_attr, default_seed))
    return bootstrap(
        seed=seed,
        env_file=env_file,
        data_subdir=data_subdir,
        checkpoint_subdir=checkpoint_subdir,
    )


def parse_and_bootstrap(
    schema: Sequence[ArgSpec],
    *,
    argv: Iterable[str] | None = None,
    description: str | None = None,
    epilog: str | None = None,
    seed_attr: str = "seed",
    default_seed: int = 42,
    env_file: str | None = None,
    data_subdir: str = "ShapeNetV2",
    checkpoint_subdir: str | None = None,
) -> tuple[argparse.Namespace, BootstrapConfig]:
    """Parse CLI arguments and bootstrap runtime configuration in one step."""
    args = parse_args(
        schema,
        argv=argv,
        description=description,
        epilog=epilog,
    )
    cfg = bootstrap_from_args(
        args,
        seed_attr=seed_attr,
        default_seed=default_seed,
        env_file=env_file,
        data_subdir=data_subdir,
        checkpoint_subdir=checkpoint_subdir,
    )
    return args, cfg


__all__ = [
    "ArgSpec",
    "build_parser",
    "parse_args",
    "bootstrap_from_args",
    "parse_and_bootstrap",
]
