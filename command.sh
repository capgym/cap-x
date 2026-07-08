uv run --no-sync --active python tools/g1_vision_bridge/client_zmq_to_capx.py \
    --robot-host 192.168.123.164 \
    --robot-port 5555 \
    --interface enx6c1ff7c1192d \
    --capx-host 127.0.0.1 \
    --capx-port 9000 \
    --intrinsics-yaml env_configs/g1/g1_d435_depth_640x480_intrinsics.yaml \
    --extrinsics-yaml env_configs/g1/g1_d435_depth_optical_extrinsics_torso.yaml

uv run --no-sync --active python tools/g1_vision_bridge/g1_grasp_bottle_preflight.py \
    --config-path env_configs/g1/g1_grasp_bottle.yaml \
    --object-name "plastic water bottle" \
    --output-dir outputs/g1_grasp_bottle_preflight

xdg-open outputs/g1_grasp_bottle_preflight/preflight_report.html


export UNITREE_NETWORK_INTERFACE=enx6c1ff7c1192d
export CAPX_G1_DRY_RUN=false

uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/g1/g1_grasp_bottle.yaml \
    --model qwen/qwen3.7-max \
    --server-url http://127.0.0.1:8110/chat/completions \
    --temperature 0.2 \
    --total-trials 1 \
    --num-workers 1 \
    --record-video True \
    --output-dir outputs/g1_grasp_bottle_real

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy=127.0.0.1,localhost
  export UNITREE_NETWORK_INTERFACE=enx6c1ff7c1192d
  export CAPX_G1_DRY_RUN=false

  uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/g1/g1_grasp_bottle.yaml \
    --model qwen/qwen3.7-max \
    --server-url http://127.0.0.1:8110/chat/completions \
    --temperature 0.2 \
    --total-trials 1 \
    --num-workers 1 \
    --record-video True \
    --output-dir outputs/g1_grasp_bottle_real

cd /home/peilab/development/cap-x

  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy=127.0.0.1,localhost
  export UNITREE_NETWORK_INTERFACE=enx6c1ff7c1192d
  export CAPX_G1_DRY_RUN=false

uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/g1/g1_grasp_bottle.yaml \
    --model qwen/qwen3.7-max \
    --server-url http://127.0.0.1:8110/chat/completions \
    --temperature 0.2 \
    --total-trials 1 \
    --num-workers 1 \
    --record-video True \
    --output-dir outputs/g1_grasp_bottle_real


export CAPX_G1_DRY_RUN=false
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

.venv/bin/python capx/envs/launch.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --model qwen/qwen3.7-max \
  --server-url http://127.0.0.1:8110/chat/completions \
  --temperature 1.0 \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True


export PARATERA_API_KEY="sk-NkiRwZEsLx5N9iW66jmT5A"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export CAPX_G1_DRY_RUN=false


uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --num-workers 1 \
  --total-trials 1 \
  --server-url https://llmapi.paratera.com/v1/chat/completions \
  --model Qwen3.6-Plus \
  --temperature 1.0 \
  --record-video True