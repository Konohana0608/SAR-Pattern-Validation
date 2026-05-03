#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./kill_voila_servers.sh [--clean-runtime-state] [--dry-run]

Options:
  --clean-runtime-state  Remove persisted Voila UI state and uploaded measured data.
  --dry-run              Print the actions without sending any signals.
EOF
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
clean_runtime_state=0
dry_run=0

for arg in "$@"; do
    case "$arg" in
        --clean-runtime-state)
            clean_runtime_state=1
            ;;
        --dry-run)
            dry_run=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

declare -A process_groups=()

while read -r pid pgid command; do
    [[ -n "${pgid:-}" ]] || continue
    process_groups["$pgid"]="$command"
done < <(
    ps -eo pid=,pgid=,command= |
        grep -E 'run_voila_smoke\.py| -m voila ' |
        grep -v grep
)

mapfile -t orphan_kernel_pids < <(
    ps -eo pid=,command= |
        grep -E 'ipykernel_launcher.+/tmp/voila_' |
        grep -v grep |
        awk '{print $1}'
)

if [[ "${#process_groups[@]}" -eq 0 && "${#orphan_kernel_pids[@]}" -eq 0 ]]; then
    echo "No stale Voila processes found."
else
    for pgid in "${!process_groups[@]}"; do
        echo "Stopping process group ${pgid}: ${process_groups[$pgid]}"
        if [[ "$dry_run" -eq 0 ]]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
    done

    if [[ "${#orphan_kernel_pids[@]}" -gt 0 ]]; then
        echo "Stopping orphaned Voila kernels: ${orphan_kernel_pids[*]}"
        if [[ "$dry_run" -eq 0 ]]; then
            kill -TERM "${orphan_kernel_pids[@]}" 2>/dev/null || true
        fi
    fi

    if [[ "$dry_run" -eq 0 ]]; then
        sleep 2
        for pgid in "${!process_groups[@]}"; do
            if ps -eo pgid= | awk -v target="$pgid" '$1 == target { found=1 } END { exit !found }'; then
                echo "Force killing remaining process group ${pgid}."
                kill -KILL -- "-$pgid" 2>/dev/null || true
            fi
        done

        if [[ "${#orphan_kernel_pids[@]}" -gt 0 ]]; then
            mapfile -t remaining_orphans < <(
                ps -eo pid=,command= |
                    grep -E 'ipykernel_launcher.+/tmp/voila_' |
                    grep -v grep |
                    awk '{print $1}'
            )
            if [[ "${#remaining_orphans[@]}" -gt 0 ]]; then
                echo "Force killing remaining orphaned Voila kernels: ${remaining_orphans[*]}"
                kill -KILL "${remaining_orphans[@]}" 2>/dev/null || true
            fi
        fi
    fi
fi

if [[ "$clean_runtime_state" -eq 1 ]]; then
    echo "Removing persisted runtime state."
    rm -f \
        "$repo_root/notebooks/system_state/ui_state.json" \
        "$repo_root/notebooks/uploaded_data/measured_data.csv"
fi

echo "Voila cleanup complete."
