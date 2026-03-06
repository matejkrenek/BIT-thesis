#!/usr/bin/env python3
"""Watch GPU availability and notify Discord when state changes.

This script is designed to run unattended (for example in tmux).
It checks which GPU compute processes are running, maps PIDs to Linux users,
and reports whether GPUs are free for a target user.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from discord_webhook import DiscordWebhook, DiscordEmbed


@dataclass
class GpuInfo:
    index: int
    uuid: str
    name: str
    mem_total_mib: int
    mem_used_mib: int


@dataclass
class GpuProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int
    owner: str
    cmd: str


def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_str()}] {message}", flush=True)


def run_cmd(args: List[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(args)}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def get_gpus() -> List[GpuInfo]:
    output = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: List[GpuInfo] = []
    if not output:
        return gpus

    for raw in output.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            GpuInfo(
                index=int(parts[0]),
                uuid=parts[1],
                name=parts[2],
                mem_total_mib=int(parts[3]),
                mem_used_mib=int(parts[4]),
            )
        )
    return gpus


def get_pid_owner_cmd(pid: int) -> Tuple[str, str]:
    output = run_cmd(["ps", "-p", str(pid), "-o", "user=,cmd="])
    if not output:
        return "<unknown>", "<unknown>"

    line = output.strip()
    if not line:
        return "<unknown>", "<unknown>"

    split = line.split(maxsplit=1)
    if len(split) == 1:
        return split[0], "<unknown>"
    return split[0], split[1]


def get_gpu_processes() -> List[GpuProcess]:
    output = run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )

    processes: List[GpuProcess] = []
    if not output:
        return processes

    for raw in output.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 4:
            continue

        gpu_uuid = parts[0]
        pid = int(parts[1])
        process_name = parts[2]
        used_memory_mib = int(parts[3])

        try:
            owner, cmd = get_pid_owner_cmd(pid)
        except Exception:
            owner, cmd = "<exited>", "<exited>"

        processes.append(
            GpuProcess(
                gpu_uuid=gpu_uuid,
                pid=pid,
                process_name=process_name,
                used_memory_mib=used_memory_mib,
                owner=owner,
                cmd=cmd,
            )
        )

    return processes


def send_discord(
    webhook_url: str,
    event_title: str,
    status: str,
    summary: Dict[str, object],
) -> None:
    color_map = {
        "FREE": "2ecc71",
        "BUSY": "e67e22",
        "ERROR": "e74c3c",
    }

    webhook = DiscordWebhook(url=webhook_url)
    embed = DiscordEmbed(
        title=event_title,
        description=f"GPU status: **{status}**",
        color=color_map.get(status, "3498db"),
    )

    embed.set_timestamp()
    embed.add_embed_field(
        name="Watcher User",
        value=f"`{summary['user']}`",
        inline=True,
    )
    embed.add_embed_field(
        name="Required Free GPUs",
        value=f"`{summary['min_free_gpus']}`",
        inline=True,
    )
    embed.add_embed_field(
        name="Currently Free",
        value=f"`{summary['available_count']}/{summary['gpu_count']}`",
        inline=True,
    )

    available_lines: List[str] = summary["available_lines"]  # type: ignore[assignment]
    blocked_lines: List[str] = summary["blocked_lines"]  # type: ignore[assignment]

    if available_lines:
        embed.add_embed_field(
            name="Available GPUs",
            value="\n".join(available_lines),
            inline=False,
        )

    if blocked_lines:
        embed.add_embed_field(
            name="Blocked By",
            value="\n".join(blocked_lines),
            inline=False,
        )

    webhook.add_embed(embed)
    response = webhook.execute()

    status_code = getattr(response, "status_code", None)
    ok = getattr(response, "ok", None)

    if status_code is not None and not (200 <= status_code < 300):
        raise RuntimeError(f"Discord returned status {status_code}")
    if ok is False:
        raise RuntimeError("Discord webhook request failed")


def summarize_state(
    user: str,
    min_free_gpus: int,
    gpus: List[GpuInfo],
    processes: List[GpuProcess],
) -> Tuple[bool, Dict[str, object]]:
    proc_by_uuid: Dict[str, List[GpuProcess]] = {}
    for process in processes:
        proc_by_uuid.setdefault(process.gpu_uuid, []).append(process)

    available: List[GpuInfo] = []
    blocked: List[Tuple[GpuInfo, List[GpuProcess]]] = []

    for gpu in gpus:
        gpu_procs = proc_by_uuid.get(gpu.uuid, [])
        foreign = [p for p in gpu_procs if p.owner != user]
        if not foreign:
            available.append(gpu)
        else:
            blocked.append((gpu, foreign))

    is_free_for_user = len(available) >= min_free_gpus

    available_lines: List[str] = []
    blocked_lines: List[str] = []

    if available:
        for gpu in available:
            available_lines.append(
                f"GPU {gpu.index} ({gpu.name}): {gpu.mem_used_mib}/{gpu.mem_total_mib} MiB"
            )

    if blocked:
        for gpu, owners in blocked:
            unique_owners = sorted({p.owner for p in owners})
            blocked_lines.append(
                f"GPU {gpu.index} ({gpu.name}): {', '.join(unique_owners)}"
            )

    summary: Dict[str, object] = {
        "user": user,
        "min_free_gpus": min_free_gpus,
        "gpu_count": len(gpus),
        "available_count": len(available),
        "available_lines": available_lines,
        "blocked_lines": blocked_lines,
    }

    return is_free_for_user, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch GPUs and send Discord messages when availability changes."
    )
    parser.add_argument(
        "--webhook-url",
        default="",
        help="Discord webhook URL. Defaults to DISCORD_WEBHOOK_URL env if omitted.",
    )
    parser.add_argument(
        "--user",
        default=getpass.getuser(),
        help="Username to evaluate GPU availability for (default: current user).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Polling interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--min-free-gpus",
        type=int,
        default=1,
        help="Required number of GPUs free from other users (default: 1).",
    )
    parser.add_argument(
        "--heartbeat-minutes",
        type=int,
        default=0,
        help="Optional heartbeat interval in minutes (0 disables).",
    )
    parser.add_argument(
        "--no-startup-message",
        action="store_true",
        help="Do not send startup status to Discord.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    webhook_url = args.webhook_url or ""

    if not webhook_url:
        import os

        webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")

    if not webhook_url:
        print(
            "Missing webhook URL. Pass --webhook-url or set DISCORD_WEBHOOK_URL.",
            file=sys.stderr,
        )
        return 2

    log("Starting GPU availability watcher")
    log(f"Target user: {args.user}")
    log(f"Polling interval: {args.interval}s")
    log(f"Required free GPUs: {args.min_free_gpus}")

    last_state: bool | None = None
    last_heartbeat = time.time()

    while True:
        try:
            gpus = get_gpus()
            processes = get_gpu_processes()
            state, summary = summarize_state(
                args.user, args.min_free_gpus, gpus, processes
            )

            if last_state is None and not args.no_startup_message:
                title = "GPU Watcher Started"
                status = "FREE" if state else "BUSY"
                send_discord(webhook_url, title, status, summary)
                log(f"Startup status sent ({status})")
                last_state = state
            elif last_state is None:
                last_state = state
                log("Startup message disabled")
            elif state != last_state:
                status = "FREE" if state else "BUSY"
                send_discord(webhook_url, "GPU Status Changed", status, summary)
                log(f"State changed -> {status}")
                last_state = state

            if args.heartbeat_minutes > 0:
                now = time.time()
                if now - last_heartbeat >= args.heartbeat_minutes * 60:
                    status = "FREE" if state else "BUSY"
                    send_discord(webhook_url, "GPU Watcher Heartbeat", status, summary)
                    log("Heartbeat sent")
                    last_heartbeat = now

        except Exception as exc:
            log(f"Watcher error: {type(exc).__name__}: {exc}")
            try:
                error_summary: Dict[str, object] = {
                    "user": args.user,
                    "min_free_gpus": args.min_free_gpus,
                    "gpu_count": 0,
                    "available_count": 0,
                    "available_lines": [],
                    "blocked_lines": [f"{type(exc).__name__}: {exc}"],
                }
                send_discord(
                    webhook_url,
                    "GPU Watcher Error",
                    "ERROR",
                    error_summary,
                )
            except Exception as send_exc:
                log(f"Failed to send error notification: {send_exc}")

        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    raise SystemExit(main())
