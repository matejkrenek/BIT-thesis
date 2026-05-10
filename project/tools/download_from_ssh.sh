#!/usr/bin/env bash

# Author: Matěj Křenek (xkrenem00)
# Contact: xkrenem00@vutbr.cz
# File: download_from_ssh.sh
# Responsibility: Downloads a remote directory over SSH to a local directory using sshpass and rsync.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

die() {
    echo "Error: $*" >&2
    exit 1
}

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        die "$cmd is required but not installed."
    fi
}

require_value() {
    local opt="$1"
    local val="${2-}"
    if [[ -z "$val" || "$val" == -* ]]; then
        die "Missing value for $opt"
    fi
}

usage() {
    cat <<'EOF'
Usage:
  download_from_ssh.sh -s <server> -p <password> -r <remote_dir> -l <local_dir>

Required arguments:
  -s, --server       SSH server in the form user@host (or host if your SSH config provides user)
  -p, --password     SSH password
  -r, --remote-dir   Remote directory to download from
  -l, --local-dir    Local directory to download to

Example:
  ./download_from_ssh.sh \
    --server user@example.com \
    --password mysecret \
    --remote-dir /home/user/data \
    --local-dir ./data_backup
EOF
}

SERVER=""
PASSWORD=""
REMOTE_DIR=""
LOCAL_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--server)
            require_value "$1" "${2-}"
            SERVER="${2:-}"
            shift 2
            ;;
        -p|--password)
            require_value "$1" "${2-}"
            PASSWORD="${2:-}"
            shift 2
            ;;
        -r|--remote-dir)
            require_value "$1" "${2-}"
            REMOTE_DIR="${2:-}"
            shift 2
            ;;
        -l|--local-dir)
            require_value "$1" "${2-}"
            LOCAL_DIR="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$SERVER" || -z "$PASSWORD" || -z "$REMOTE_DIR" || -z "$LOCAL_DIR" ]]; then
    echo "Error: missing required arguments." >&2
    usage
    exit 1
fi

require_command sshpass
require_command rsync

mkdir -p "$LOCAL_DIR"

# Use rsync for robust recursive transfer and resume support.
sshpass -p "$PASSWORD" rsync \
    -az --info=progress2 \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    "$SERVER:$REMOTE_DIR/" "$LOCAL_DIR/"

echo "Download completed: $SERVER:$REMOTE_DIR -> $LOCAL_DIR"
