#!/bin/bash
set -eu

INSTALL_DIR=${MLSUB_GIT_WRAPPER_DIR:-"/home/jovyan/mlsub-bin"}
REAL_GIT=${MLSUB_REAL_GIT:-"/usr/bin/git"}

if [ "$(basename "$0")" != "git" ]; then
    mkdir -p "$INSTALL_DIR"
    cp "$0" "$INSTALL_DIR/git"
    chmod +x "$INSTALL_DIR/git"
    cp "$(dirname "$0")/mlsub_bootstrap_env.sh" /home/jovyan/mlsub_bootstrap_env.sh
    echo "Installed mlsub git wrapper at $INSTALL_DIR/git"
    exit 0
fi

if [ "${1:-}" = "clone" ] \
    && [ "${OMPI_COMM_WORLD_SIZE:-1}" -gt 1 ] \
    && [ "${!#}" = "/tmp/app" ]; then
    READY_FILE=/tmp/mlsub_git_clone_ready
    if [ "${OMPI_COMM_WORLD_RANK:-0}" = "0" ]; then
        rm -f "$READY_FILE"
        sleep 2
        "$REAL_GIT" "$@"
        touch "$READY_FILE"
        exit 0
    fi

    for _ in $(seq 1 600); do
        if [ -f "$READY_FILE" ]; then
            exit 0
        fi
        sleep 0.5
    done
    echo "Timed out waiting for rank 0 to clone /tmp/app" >&2
    exit 1
fi

exec "$REAL_GIT" "$@"
