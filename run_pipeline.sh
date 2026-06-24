#!/usr/bin/env bash
# ============================================
# PPO Trading Pipeline — Bash Entrypoint
# For cron / systemd usage on Linux
# ============================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ──────────────────────────────────
CONDA_ENV="${PIPELINE_CONDA_ENV:-dl_final}"
# Auto-detect conda if not explicitly set
if [ -z "${PIPELINE_CONDA_PATH:-}" ]; then
    CONDA_EXE="$(command -v conda 2>/dev/null || true)"
    if [ -n "$CONDA_EXE" ]; then
        CONDA_PATH="$(dirname "$(dirname "$CONDA_EXE")")"
    else
        # Fallback search paths
        for p in /opt/conda /usr/local/anaconda3 ~/anaconda3 ~/miniconda3 /usr/share/miniconda3; do
            if [ -f "$p/etc/profile.d/conda.sh" ]; then
                CONDA_PATH="$p"
                break
            fi
        done
    fi
fi
CONDA_PATH="${CONDA_PATH:-/opt/conda}"
PYTHON="${CONDA_PATH}/envs/${CONDA_ENV}/bin/python"
PIDFILE="/tmp/ppo_pipeline.pid"
LOCKFILE="/tmp/ppo_pipeline.lock"

# ── Help ────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
  fetch       Download latest API data
  gru         Recompute GRU cache for updated stocks
  trade       Execute PPO trading
  daily       fetch + gru + reconcile (post-market update)
  full        fetch + gru + trade (full pipeline)
  status      Show pipeline state
  train-gru   GRU pre-training
  train-ppo   PPO reinforcement learning training
  bot         Start Discord bot (foreground)
  --help      This help

Options:
  --simulate  Dry-run trades (no real API orders)

Environment:
  PIPELINE_CONDA_ENV  Conda env name (default: dl_final)
  PIPELINE_CONDA_PATH Conda root (default: /opt/conda)

Cron examples:
  # Post-market data + GRU + reconcile at 16:30 Mon-Fri
  30 16 * * 1-5 /path/to/run_pipeline.sh daily >> /path/to/logs/cron.log 2>&1
EOF
    exit 0
}

# ── Lock ────────────────────────────────────
acquire_lock() {
    if ! mkdir "$LOCKFILE" 2>/dev/null; then
        echo "[$(date)] ERROR: Another instance running (lock: $LOCKFILE)" >&2
        exit 1
    fi
    trap 'rmdir "$LOCKFILE" 2>/dev/null || true' EXIT
}

# ── PID file ────────────────────────────────
write_pid() {
    echo $$ > "$PIDFILE"
}

# ── Check conda env ─────────────────────────
check_env() {
    if [ ! -f "$PYTHON" ]; then
        echo "[$(date)] ERROR: Python not found: $PYTHON" >&2
        echo "  Set PIPELINE_CONDA_PATH/PIPELINE_CONDA_ENV or install conda" >&2
        exit 1
    fi
}

# ── Main ────────────────────────────────────
main() {
    local cmd="${1:-help}"
    shift 2>/dev/null || true
    local simulate=""

    # Parse common options
    while [ $# -gt 0 ]; do
        case "$1" in
            --simulate) simulate="--simulate"; shift ;;
            --help) usage ;;
            *) break ;;
        esac
    done

    case "$cmd" in
        fetch|gru|trade|daily|full|status|reconcile|train-gru)
            acquire_lock
            write_pid
            check_env
            exec "$PYTHON" "$SCRIPT_DIR/main.py" "$cmd" $simulate
            ;;
        train-ppo|batch-sim)
            acquire_lock
            write_pid
            check_env
            exec "$PYTHON" "$SCRIPT_DIR/main.py" "$cmd" "$@"
            ;;
        bot)
            check_env
            exec "$PYTHON" "$SCRIPT_DIR/main.py" bot
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo "[$(date)] Unknown command: $cmd" >&2
            usage
            ;;
    esac
}

main "$@"
