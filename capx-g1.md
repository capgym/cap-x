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

--use-oracle-code True

Homie+Gateway 启动方式

先 dry-run：

  # 终端 1：Gateway dry-run，不发布 rt/lowcmd
  /home/peilab/development/WholeBodyGateway/robot_control_gateway/build-g1/bin/humanoid_g1_control_gateway \
    enx6c1ff7c1192d \
    --dry-run

    --arm-timeout-mode damping

  # 终端 2：OpenHomie 下肢 policy
  conda activate homie310
  cd /home/peilab/development/OpenHomie/HomieDeploy
  python g1_gym_deploy/scripts/deploy_policy.py

  # 终端 3：WholeBodyGateway 上肢/腰/下肢 API dry-run
  conda activate unitree
  cd /home/peilab/development/WholeBodyGateway
  python -m upper_body_policy.manual_whole_body_api \
    --upper-policy-dry-run \
    --max-joint-step 0.012 \
    --interactive

  真机输出时，把终端 1 的 --dry-run 去掉；终端 3 改成：
    --real-lower \
    --real-upper \
    --upper-policy-root /home/peilab/development/upper_policy \
    --startup-blend-time 4.0 \
    --target-smoothing-alpha 0.65 \
    --max-joint-step 0.012 \
    --interactive

先deactivate capx的venv




## G1 站立 + CaP-X 抓取四终端启动顺序

前提：机器人相机服务已启动；G1 与本机通过 `enx6c1ff7c1192d` 联通；现场有人看护急停。协同模式下只有 Gateway 发布 `rt/lowcmd`，不要同时启动其它直发上肢/全身控制程序。

### 终端 1：启动 WholeBodyGateway

不需要 conda。这个 Gateway 保持下肢 Homie 控制，上肢 `arm_action` 超时后进入阻尼，不再保持最后抓取姿态。

```bash
cd /home/peilab/development/cap-x

/home/peilab/development/WholeBodyGateway/robot_control_gateway/build-g1-v2/bin/humanoid_g1_control_gateway \
  enx6c1ff7c1192d \
  --arm-timeout-mode damping
```

### 终端 2：启动 OpenHomie 下肢站立/平衡策略

```bash
conda activate homie310
cd /home/peilab/development/OpenHomie/HomieDeploy

python g1_gym_deploy/scripts/deploy_policy.py
```

按 OpenHomie 提示完成 G1 站立。确认 Gateway 收到下肢命令后，再启动上肢抓取。

### 终端 3：启动 G1 图像桥接和 SAM3 viewer

先启动图像服务。此时 CaP-X launch 还没有开启 9000 observation server，所以会看到 `Waiting for 127.0.0.1:9000`，这是正常的；终端 4 启动后会自动连上。

```bash
cd /home/peilab/development/cap-x
source .venv/bin/activate

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
```

Viewer 地址：`http://127.0.0.1:9010/`。按 `R` 可以重新抓取最新图像、重新跑 SAM3，并显示新的抓取点位。

### 终端 4：启动 CaP-X 抓取流程

启动后会开启 8114/8115/8116 服务和 9000 observation server；终端 3 的图像桥会自动连接进来。

```bash
cd /home/peilab/development/cap-x
source .venv/bin/activate

export CAPX_G1_DRY_RUN=false
export NO_PROXY=127.0.0.1,localhost,192.168.123.0/24
export no_proxy=127.0.0.1,localhost,192.168.123.0/24

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --num-workers 1 \
  --total-trials 1 \
  --record-video True \
  --use-oracle-code True
```
