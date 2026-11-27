#!/usr/bin/env bash
set -euo pipefail

gpu_request=${1:-auto}
startup_seconds=${CAPX_A800_STARTUP_SECONDS:-480}
max_model_len=${CAPX_A800_MAX_MODEL_LEN:-32768}
if (( $# > 2 )) || [[ $gpu_request != auto && ! $gpu_request =~ ^[0-9]+$ ]] ||
    [[ ! $startup_seconds =~ ^[1-9][0-9]*$ ]] || [[ ! $max_model_len =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: bash start_a800.sh [auto|GPU] [A800_BASE]\n' >&2
    exit 2
fi
base=$(cd -- "${2:-${CAPX_A800_BASE:-$HOME}}" && pwd -P)

export WORK_DIR=$base/pear_official_runtime
server_env=$WORK_DIR/envs/server
model=$base/models/Qwen3.8-27B
runtime=$base/capx-runtime
test -x "$server_env/bin/vllm"
test -f "$model/model.safetensors.index.json"
mkdir -p "$runtime"

idle_gpu() {
    local memory total used compute
    memory=$(nvidia-smi -i "$1" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits) || return 1
    IFS=, read -r total used <<< "$memory"
    compute=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader) || return 1
    [[ $total -ge 78000 && $used -lt 256 && -z $compute ]]
}

gpu=$gpu_request
if [[ $gpu == auto ]]; then
    gpu=
    indexes=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
    while read -r candidate; do
        if idle_gpu "$candidate"; then gpu=$candidate; break; fi
    done <<< "$indexes"
fi
if [[ -z $gpu ]] || ! idle_gpu "$gpu"; then
    printf 'No requested idle GPU with at least 78,000 MiB available for Qwen3.8-27B.\n' >&2
    exit 1
fi

port=${CAPX_A800_PORT:-}
if [[ -z $port ]]; then
    port=$("$server_env/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
fi
if [[ ! $port =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
    printf 'CAPX_A800_PORT must be in 1–65535.\n' >&2
    exit 2
fi

cache=${CAPX_A800_CACHE_DIR:-$runtime/qwen27b-graph-mtp-cache}
mkdir -p -- "$cache"
test -w "$cache"

run_dir=$(mktemp -d "$runtime/qwen27b-mm-mtp.XXXXXXXX")
export LOCAL_DIR=$cache
export TMPDIR=$run_dir/tmp
export XDG_CACHE_HOME=$LOCAL_DIR/xdg TORCH_HOME=$LOCAL_DIR/torch
export HF_HOME=$WORK_DIR/cache/hf PYTHONPYCACHEPREFIX=$LOCAL_DIR/pycache
mkdir -p "$TMPDIR"
compat=$("$server_env/bin/python" -c 'import json,os; print(json.load(open(os.environ["WORK_DIR"]+"/runtime.json"))["cuda_compat_dir"])')
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_ENABLE_CUDA_COMPATIBILITY=1 VLLM_CUDA_COMPATIBILITY_PATH=$compat
source "$WORK_DIR/src/pear/scripts/gpu_process_variables.sh" "$server_env" >&2
export CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_PLUGINS=
export CC=/usr/bin/clang-12 CXX=/usr/bin/clang++-12
export VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False
unset PYTORCH_CUDA_ALLOC_CONF PEAR_DEPLOY_KERNELS PEAR_ROLLOUT_KERNELS
export CAPX_QWEN27B_OWNER=$run_dir

listener=$(ss -H -ltn "sport = :$port")
if ! idle_gpu "$gpu" || [[ -n $listener ]]; then
    printf 'A800 GPU %s or port %s became occupied during setup.\n' "$gpu" "$port" >&2
    exit 1
fi

pid=
cleanup_failed_start() {
    local status=$?
    if [[ $status -ne 0 && -n $pid ]] && grep -zFxq "CAPX_QWEN27B_OWNER=$run_dir" "/proc/$pid/environ" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        printf 'Stopped owned startup PID %s. See %s/server.log.\n' "$pid" "$run_dir" >&2
    fi
}
trap cleanup_failed_start EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT

setsid nohup "$server_env/bin/vllm" serve "$model" \
    --host 127.0.0.1 --port "$port" --served-model-name Qwen/Qwen3.8-27B \
    --dtype bfloat16 --tensor-parallel-size 1 --generation-config vllm \
    --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking":false}' \
    --limit-mm-per-prompt '{"image":{"count":2,"width":640,"height":480},"video":0}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --gpu-memory-utilization 0.83 --max-model-len "$max_model_len" --max-num-seqs 2 \
    >"$run_dir/server.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$run_dir/server.pid"
printf 'Waiting for A800 vLLM PID %s on GPU %s with a %s-token context. Log %s/server.log\n' \
    "$pid" "$gpu" "$max_model_len" "$run_dir" >&2

deadline=$((SECONDS + startup_seconds))
while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null || [[ $(ps -o stat= -p "$pid") == Z* ]]; then break; fi
    if [[ $(ss -H -ltnp "sport = :$port") == *"pid=$pid,"* ]] &&
        curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 &&
        curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null |
            "$server_env/bin/python" -c 'import json,sys; sys.exit(not any(m["id"]=="Qwen/Qwen3.8-27B" for m in json.load(sys.stdin)["data"]))' 2>/dev/null; then
        printf 'RUN_DIR=%s PID=%s PORT=%s GPU=%s\n' "$run_dir" "$pid" "$port" "$gpu"
        exit 0
    fi
    sleep 2
done
printf 'A800 vLLM did not become healthy within %s seconds.\n' "$startup_seconds" >&2
tail -n 24 "$run_dir/server.log" >&2
exit 1
