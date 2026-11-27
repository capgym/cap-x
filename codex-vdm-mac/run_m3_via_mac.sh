#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trials="${1:-1}"
[[ "$trials" =~ ^[1-9][0-9]*$ ]] || { echo "Usage: CAPX_REMOTE_REPO=/absolute/remote/checkout bash run_m3_via_mac.sh [trials]" >&2; exit 2; }

config="${CAPX_M3_CONFIG:-env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml}"
codex_cli="${CODEX_BIN:-/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex}"
ssh_host="${CAPX_SSH_HOST:-4090w}"
remote_repo="${CAPX_REMOTE_REPO:-}"
remote_uv="${CAPX_REMOTE_UV:-}"
local_port=8117

if [[ ! "$ssh_host" =~ ^([a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]]; then
    echo "Set CAPX_SSH_HOST to an SSH alias or user@hostname." >&2
    exit 2
fi
if [[ "$remote_repo" != /* || "$remote_repo" == *$'\n'* || "$remote_repo" == *$'\r'* ]]; then
    echo "CAPX_REMOTE_REPO must be an absolute path on the remote host." >&2
    exit 2
fi
if [[ -n "$remote_uv" && ( "$remote_uv" != /* || "$remote_uv" == *$'\n'* || "$remote_uv" == *$'\r'* ) ]]; then
    echo "CAPX_REMOTE_UV must be an absolute path to the remote uv executable." >&2
    exit 2
fi
if [[ "$config" == *$'\n'* || "$config" == *$'\r'* ]]; then
    echo "CAPX_M3_CONFIG must be a single-line path." >&2
    exit 2
fi
remote_uv="${remote_uv:-uv}"

if [[ ! -x "$codex_cli" ]]; then
    echo "Codex CLI is missing. Set CODEX_BIN to the executable on your Mac." >&2
    exit 1
fi
if ! "$codex_cli" login status 2>&1 | grep -qi 'Logged in using ChatGPT'; then
    echo "Sign in on your Mac first: \"$codex_cli\" login" >&2
    exit 1
fi

token="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
run_id="m3_stack_codex_mac_$(TZ=Asia/Taipei date +%Y%m%d_%H%M%S)"
trace_code="$(python3 -c 'import base64,sys; print(base64.b64encode(open(sys.argv[1], "rb").read()).decode())' "$script_dir/dashboard_instrumentation.py")"
dashboard_dir="$(python3 "$script_dir/dashboard.py" run-dir "$run_id")"
control_dir="$(mktemp -d /tmp/capx-vdm.XXXXXXXX)"
control_socket="$control_dir/ssh"
ssh_connection=(ssh -S "$control_socket" -o ConnectTimeout=10)

python3 "$script_dir/codex_vdm_server.py" "$codex_cli" "$local_port" <<< "$token" &
server_pid=$!

cleanup() {
    local status=$?
    python3 "$script_dir/dashboard.py" finish "$run_id" "$status" || true
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    "${ssh_connection[@]}" -O exit "$ssh_host" >/dev/null 2>&1 || true
    rm -f "$control_socket"
    rmdir "$control_dir"
}
trap cleanup EXIT

ready=false
for ((attempt = 0; attempt < 30; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "The local VDM server could not start." >&2
        exit 1
    fi
    if curl --noproxy '*' -fsS --max-time 1 "http://127.0.0.1:$local_port/health" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 0.1
done
if [[ "$ready" != true ]]; then
    echo "The local VDM server did not respond." >&2
    exit 1
fi

printf 'Connecting to %s for checkout %s.\n' "$ssh_host" "$remote_repo"
"${ssh_connection[@]}" -fNT -M -o ControlPersist=no \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$ssh_host"
remote_port="$("${ssh_connection[@]}" -O forward -o ExitOnForwardFailure=yes \
    -R "127.0.0.1:0:127.0.0.1:$local_port" "$ssh_host")"
if [[ ! "$remote_port" =~ ^[1-9][0-9]{0,4}$ ]]; then
    echo "SSH did not return an allocated VDM port." >&2
    exit 1
fi

remote_command="$(cat <<'REMOTE'
set -eu
IFS= read -r CAPX_CODEX_VDM_TOKEN
IFS= read -r trials
IFS= read -r config
IFS= read -r remote_port
IFS= read -r run_id
IFS= read -r CAPX_DASHBOARD_CODE
IFS= read -r remote_repo
IFS= read -r remote_uv
export CAPX_CODEX_VDM_TOKEN CAPX_DASHBOARD_CODE

cd "$remote_repo"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v "$remote_uv" >/dev/null 2>&1; then
    printf 'Could not find remote uv executable %s. Set CAPX_REMOTE_UV on the Mac.\n' "$remote_uv" >&2
    exit 1
fi
curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$remote_port/health"
printf '\nSSH tunnel to Mac Codex is ready on remote port %s.\n' "$remote_port"

launcher="$(cat <<'PY'
import base64
import os

events = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)

import tyro

from capx.envs.launch import LaunchArgs, main
from capx.llm.client import VLM_MODELS

VLM_MODELS.append("openai/codex-cli-vdm")
args = tyro.cli(LaunchArgs)
args.visual_differencing_model_api_key = os.environ.pop("CAPX_CODEX_VDM_TOKEN")
tracing = {}
exec(base64.b64decode(os.environ.pop("CAPX_DASHBOARD_CODE")), tracing)
tracing["install_dashboard_tracing"](args, stream=events)
main(args)
PY
)"

"$remote_uv" run --no-sync python -u -c "$launcher" \
    --config-path "$config" \
    --use-oracle-code False \
    --server-url http://127.0.0.1:18089/v1/chat/completions \
    --model checkpoints/Qwen2.5-Coder-7B-Instruct \
    --visual-differencing-model openai/codex-cli-vdm \
    --visual-differencing-model-server-url "http://127.0.0.1:$remote_port/v1/chat/completions" \
    --temperature 0.2 \
    --max-tokens 2048 \
    --total-trials "$trials" \
    --num-workers 1 \
    --output-dir "outputs/$run_id"
REMOTE
)"

set +e
printf '%s\n' "$token" "$trials" "$config" "$remote_port" "$run_id" "$trace_code" "$remote_repo" "$remote_uv" |
    "${ssh_connection[@]}" -T "$ssh_host" "$remote_command" |
    python3 "$script_dir/dashboard.py" record "$run_id"
pipeline_status=("${PIPESTATUS[@]}")
set -e

if [[ "${pipeline_status[2]}" != 0 ]]; then
    echo "Could not save the dashboard events." >&2
    exit "${pipeline_status[2]}"
fi
if [[ "${pipeline_status[1]}" != 0 ]]; then
    exit "${pipeline_status[1]}"
fi

remote_run_dir="$remote_repo/outputs/checkpoints_Qwen2.5-Coder-7B-Instruct/$run_id"
copy_command="$(cat <<'REMOTE'
set -eu
IFS= read -r remote_run_dir
cd "$remote_run_dir"
find . -type f \( -name '*.mp4' -o -name 'summaries.txt' \) -print0 | tar --null -T - -cf -
REMOTE
)"
if ! printf '%s\n' "$remote_run_dir" |
    "${ssh_connection[@]}" -T "$ssh_host" "$copy_command" |
    tar -xf - -C "$dashboard_dir"; then
    echo "Could not copy the final video and summary from $ssh_host to $dashboard_dir." >&2
fi
