#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_tron2_route2.sh VENDOR_DIR TEST_OPT_DIR OUTPUT_DIR [SWEEP_ARGS]
       bash scripts/run_tron2_route2.sh VENDOR_DIR TEST_OPT_DIR OUTPUT_DIR --tier TIER [MODEL_ARGS]

Run a recorded CGN sweep with the official controller and one Python MuJoCo
world in a private Linux namespace. OUTPUT_DIR must be new.

VENDOR_DIR   Controller run pack containing Tron2_controller/ and joystick_sim/.
TEST_OPT_DIR Directory containing the simulated controller's test /opt fixture.

Optional env vars: CUDA_VISIBLE_DEVICES, TRON2_PYTHON (default .venv/bin/python),
TRON2_SDK_DIR (directory containing limxsdk if it is outside the venv).
SWEEP_ARGS can include --fresh-trials 1, --ranks 0 1, --jaw-branch 0, or
--candidate-file FILE.
TIER is S1–S4 (single-turn) or M1–M4 (multi-turn).
MODEL_ARGS include --server-url http://127.0.0.1:18089/v1/chat/completions,
--trials 1, --first-seed 1, --cube-xy-jitter-m 0.0, --max-tokens 1024, and
--max-code-blocks 4. S2 and M1–M3 can also use --candidate-file for batch replay.
--model defaults to checkpoints/Qwen2.5-Coder-7B-Instruct. M2 requires an
image-capable coding model. M3/M4 with Qwen-Coder require a separate image
differencing model at --vdm-model MODEL and --vdm-server-url URL.
For custom model IDs, pass --model-is-vlm or --vdm-model-is-vlm as appropriate.
Run the model servers on the host loopback before launching. Optional credentials
are read from TRON2_MODEL_API_KEY and TRON2_VDM_API_KEY.
USAGE
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# < 3 )); then
    usage >&2
    exit 2
fi
if [[ $(uname -s) != Linux ]]; then
    printf 'This launcher requires Linux with unprivileged user and network namespaces.\n' >&2
    exit 2
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
vendor=$(cd -- "$1" && pwd -P)
test_opt=$(cd -- "$2" && pwd -P)
output=$(realpath -m -- "$3")
shift 3
python=${TRON2_PYTHON:-$repo/.venv/bin/python}
python=$(realpath -ms -- "$python")
sdk=${TRON2_SDK_DIR:-}
if [[ -n $sdk ]]; then
    sdk=$(cd -- "$sdk" && pwd -P)
fi
for path in "$python" "$vendor/Tron2_controller/install/run_manipulation_sim.sh" \
    "$vendor/joystick_sim/pub.sh"; do
    if [[ ! -x $path ]]; then
        printf 'Executable required: %s\n' "$path" >&2
        exit 2
    fi
done
if [[ ! -f $repo/capx/assets/tron2/DACH_TRON2A/xml/robot_grasper.xml ]]; then
    printf 'Missing staged TRON2 model assets. See docs/tron2.md.\n' >&2
    exit 2
fi

# Resolve filesystem arguments before the namespace launcher changes directory.
sweep_args=()
tier=
model_url=http://127.0.0.1:18089/v1/chat/completions
vdm_url=
while (( $# )); do
    case "$1" in
        --tier|--server-url|--vdm-server-url)
            if (( $# < 2 )); then
                printf 'A value is required after %s.\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --tier) tier=$2 ;;
                --server-url) model_url=$2 ;;
                --vdm-server-url) vdm_url=$2 ;;
            esac
            shift 2
            ;;
        --candidate-file)
            if (( $# < 2 )); then
                printf 'A path is required after --candidate-file.\n' >&2
                exit 2
            fi
            sweep_args+=("$1" "$(realpath -e -- "$2")")
            shift 2
            ;;
        *)
            sweep_args+=("$1")
            shift
            ;;
    esac
done

relay_dir=
relay_socket=
vdm_relay_socket=
relay_pids=()
namespace_prefix=()
if [[ -n $tier ]]; then
    if [[ ! $tier =~ ^[SM][1-4]$ ]]; then
        printf 'The TRON2 model tier must be S1–S4 or M1–M4.\n' >&2
        exit 2
    fi
    if [[ -n $vdm_url && $tier != M3 && $tier != M4 ]]; then
        printf 'An image differencing endpoint is available for M3 and M4.\n' >&2
        exit 2
    fi
    relay_dir=$(mktemp -d /tmp/capx-tron2-relay.XXXXXX)
    relay_socket=$relay_dir/model.sock
    cleanup_relay() {
        local pid
        for pid in "${relay_pids[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        done
        rm -f -- "$relay_dir/model.sock" "$relay_dir/vdm.sock"
        rmdir -- "$relay_dir"
    }
    trap cleanup_relay EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    start_relay() {
        local relay_path=$1 server_url=$2 pid
        "$python" "$repo/scripts/run_tron2_model.py" --serve-relay "$relay_path" --server-url "$server_url" &
        pid=$!
        relay_pids+=("$pid")
        for _ in {1..50}; do
            [[ -S $relay_path ]] && return
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.1
        done
        printf 'Failed to start the relay for %s.\n' "$server_url" >&2
        return 1
    }
    start_relay "$relay_socket" "$model_url"
    if [[ -n $vdm_url ]]; then
        vdm_relay_socket=$relay_socket
        if [[ $vdm_url != "$model_url" ]]; then
            vdm_relay_socket=$relay_dir/vdm.sock
            start_relay "$vdm_relay_socket" "$vdm_url"
        fi
    fi
else
    # Preserve sweep launcher behavior and signal ownership.
    namespace_prefix=(exec)
fi

"${namespace_prefix[@]}" unshare -Urnmpi --fork --mount-proc --propagation private bash -s -- \
    "$repo" "$vendor" "$test_opt" "$output" "$python" "$sdk" "$tier" "$relay_socket" "$model_url" \
    "$vdm_relay_socket" "$vdm_url" \
    "${sweep_args[@]}" <<'NAMESPACE'
set -euo pipefail
repo=$1
vendor=$2
test_opt=$3
output=$4
python=$5
sdk=$6
tier=$7
relay_socket=$8
model_url=$9
vdm_relay_socket=${10}
vdm_url=${11}
shift 11

mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
cd -- "$repo"
ulimit -c 0
ip link set lo up
mount --bind "$test_opt" /opt
mount -t tmpfs -o mode=1777,size=512m tmpfs /dev/shm

export IS_SIM=1 MROS_LOCALHOST_ONLY=1 MROS_SIM_TIME=0
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$repo${sdk:+:$sdk}"
vendor_env=()
legacy_runtime="$vendor/.runtime/root"
if [[ -d $legacy_runtime/usr/lib/x86_64-linux-gnu ]]; then
    vendor_env+=("LD_LIBRARY_PATH=$legacy_runtime/usr/lib/x86_64-linux-gnu:$legacy_runtime/lib/x86_64-linux-gnu"
                 "TRON2_KEEP_EXTERNAL_LD_LIBRARY_PATH=1")
fi

children=()
finish() {
    status=$?
    for pid in "${children[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    if (( status != 0 )); then
        printf 'Route 2 exited with status %s. See logs in %s\n' "$status" "$output" >&2
        for log in tabletop.log controller.log joystick.log; do
            if [[ -f $output/$log ]]; then
                printf '\n%s\n' "$log" >&2
                tail -n 12 -- "$output/$log" >&2
            fi
        done
    fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for() {
    local timeout=$1 pid=$2 description=$3
    shift 3
    local deadline=$((SECONDS + timeout))
    until "$@"; do
        if (( SECONDS >= deadline )) || ! kill -0 "$pid" 2>/dev/null; then
            printf 'Failed waiting for %s.\n' "$description" >&2
            return 1
        fi
        sleep 0.1
    done
}

controller_listening() {
    [[ -n $(ss -H -ltn 'sport = :5000') ]]
}

printf 'Route 2 logs and videos: %s\n' "$output"
if [[ -n $tier ]]; then
    vdm_args=()
    if [[ -n $vdm_url ]]; then
        vdm_args=(--vdm-server-url "$vdm_url" --vdm-relay-socket "$vdm_relay_socket")
    fi
    "$python" scripts/run_tron2_model.py "$@" --tier "$tier" --relay-socket "$relay_socket" \
        "${vdm_args[@]}" \
        --server-url "$model_url" --output-dir "$output/evaluation" \
        --route2-startup-dir "$output" >"$output/tabletop.log" 2>&1 &
else
    "$python" scripts/sweep_tron2_cgn.py "$@" \
        --config-path env_configs/tron2/tron2_cube_lift_visual_cgn_only.yaml \
        --output-dir "$output/evaluation" --route2-startup-dir "$output" \
        >"$output/tabletop.log" 2>&1 &
fi
tabletop=$!
children+=("$tabletop")
wait_for 120 "$tabletop" 'MuJoCo/MROS startup' test -f "$output/motor.ready"

(
    cd -- "$vendor/Tron2_controller"
    exec env "${vendor_env[@]}" ./install/run_manipulation_sim.sh
) >"$output/controller.log" 2>&1 &
controller=$!
children+=("$controller")
wait_for 20 "$controller" 'the controller WebSocket' controller_listening
sleep 1
env "${vendor_env[@]}" "$vendor/joystick_sim/pub.sh" switch >"$output/joystick.log" 2>&1
sleep 1
touch "$output/controller.ready"
wait "$tabletop"
NAMESPACE
