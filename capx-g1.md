我已经改了 capx/serving/launch_servers.py:49，现在默认 SAM3 启动参数会自动带：

  --checkpoint-path capx/model_weights/sam3/sam3.pt

  以后直接用原命令即可：

  uv run --no-sync --active capx/serving/launch_servers.py --profile default

  但你现在手动启动的 SAM3 还占着 8114 端口，先在那个终端 Ctrl-C 停掉，再运行 default launcher。

  验证过两项：

  .venv/bin/python -m unittest tests.test_launch_servers_sam3_checkpoint tests.test_sam3_server_checkpoint -v

  结果：Ran 3 tests ... OK

  直接打印 default profile 生成的 SAM3 命令，现在是：

  /home/peilab/development/cap-x/.venv/bin/python -m capx.serving.launch_sam3_server --port 8114 --host 127.0.0.1 --device
  cuda --checkpoint-path capx/model_weights/sam3/sam3.pt



outputs/g1_grasp_bottle_fail/grasp_debug/plastic_water_bottle_grasp_debug.html
可视化了rgbd点云，发现g1 PC发送rgb和深度图像时没有对齐，导致sam3分割出的深度区域错误，因此生成抓取总是失败



export PARATERA_API_KEY="sk-NkiRwZEsLx5N9iW66jmT5A"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export CAPX_G1_DRY_RUN=false

uv run --no-sync --active python tools/g1_vision_bridge/client_zmq_to_capx.py \
    --robot-host 192.168.123.164 \
    --robot-port 5555 \
    --interface enx6c1ff7c1192d \
    --capx-host 127.0.0.1 \
    --capx-port 9000 \
    --intrinsics-yaml env_configs/g1/g1_d435_depth_640x480_intrinsics.yaml \
    --extrinsics-yaml env_configs/g1/g1_d435_depth_optical_extrinsics_torso.yaml \
    --viewer \
    --viewer-port 9010 \
    --viewer-sam3 \
    --viewer-sam3-prompt "plastic water bottle" \
    --viewer-sam3-url http://127.0.0.1:8114



export CAPX_G1_DRY_RUN=false
  export OPENAI_API_KEY="sk-NkiRwZEsLx5N9iW66jmT5A"

  uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/g1/g1_grasp_bottle.yaml \
    --num-workers 1 \
    --total-trials 1 \
    --server-url https://llmapi.paratera.com/v1/chat/completions \
    --model Qwen3.6-Plus \
    --temperature 1.0 \
    --record-video True