# Unitree G1 真机右臂配置

这个配置只用于 Unitree G1 真机右臂控制，不需要安装 Robosuite、LIBERO、Isaac
或其他仿真器相关环境。

## 已安装的 SDK

Unitree SDK2 Python 源码目录应位于：

```bash
capx/third_party/unitree_sdk2_python
```

安装到当前 CaP-X 虚拟环境：

```bash
uv --cache-dir /home/peilab/development/cap-x/.uv-cache pip install -p .venv/bin/python -e capx/third_party/unitree_sdk2_python
```

## 真机运行环境变量

设置可以连到 G1 DDS 网络的网卡：

```bash
export UNITREE_NETWORK_INTERFACE=<G1 DDS interface>
export CAPX_G1_DRY_RUN=false
```

必须设置 `UNITREE_NETWORK_INTERFACE` 为当前连接 G1 DDS 网段的网卡名。
如果没有设置 `CAPX_G1_DRY_RUN`，`G1RealLowLevel`
会默认保持 dry-run 模式，避免误发真机命令。

## CaP-X 启动方式

使用 Paratera 的 OpenAI-compatible 接口时，不要把 key 写进命令或配置文件，先在 shell 里设置环境变量：

```bash
export PARATERA_API_KEY=<your-paratera-api-key>
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

然后启动 G1 grasp bottle 流程：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --num-workers 1 \
  --total-trials 1 \
  --server-url https://llmapi.paratera.com/v1/chat/completions \
  --model Qwen3.6-Plus \
  --temperature 1.0 \
  --record-video True
```

可用模型名按 Paratera provider 配置填写，例如：`DeepSeek-V4-Pro`、`GLM-5.1`、`MiniMax-M2.7`、`Qwen3.6-Plus`。
`--server-url` 需要指向 OpenAI-compatible 的完整 `/chat/completions` 接口；如果 Paratera 账号侧要求不带 `/v1`，把它改成 `https://llmapi.paratera.com/chat/completions`。

## 任务配置

当前右臂塑料水瓶抓取任务位于：

```text
capx/envs/tasks/unitree_g1/grasp_bottle.py
env_configs/g1/g1_grasp_bottle.yaml
```

注册名是 `unitree_g1_grasp_bottle_code_env`。启动方式：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --num-workers 1
```

当前配置里的 PyRoKi server 使用：

```yaml
robot: env_configs/g1/g1_29dof_with_hand.urdf
target_link: right_hand_palm_link
```

现阶段只把右臂当作 7-DoF 单臂机器人使用，不在模型可调用 API 里区分左右臂，也不控制手指关节。
PyRoKi 使用带手的 29-DoF URDF 做 IK/FK 依据；控制输出仍只下发右臂 7 个关节。

## 可调用控制接口

`G1RealControlApi` 暴露以下函数：

- `goto_pose(position, quaternion_wxyz, z_approach=0.0)`
- `move_to_joints(joints)`
- `open_gripper()`
- `close_gripper()`：三指配合抓取，thumb/index/middle 同时闭合
- `close_index_pinch()`：大拇指转向食指，作为二维夹爪闭合
- `move_hand_joints(joints)`：直接下发右手 7 个语义关节
- 继承的视觉辅助函数：`get_object_pose()` 和 `sample_grasp_pose()`

`move_to_joints(joints)` 接受 7 个关节角，不包含手爪，顺序是：
`right_shoulder_pitch_joint`、`right_shoulder_roll_joint`、
`right_shoulder_yaw_joint`、`right_elbow_joint`、`right_wrist_roll_joint`、
`right_wrist_pitch_joint`、`right_wrist_yaw_joint`。

Dex3 右手关节在 API 中使用语义顺序：`thumb_0`、`thumb_1`、`thumb_2`、`index_0`、`index_1`、`middle_0`、`middle_1`。底层发送到 Unitree DDS 前会自动转换为真机电机顺序：`thumb_0`、`thumb_1`、`thumb_2`、`middle_0`、`middle_1`、`index_0`、`index_1`。

## 视觉输入和 API 封装

G1 相机 server 的 ZMQ/msgpack/JPEG 协议已经按 Franka API 的方式封装到：

```text
capx/integrations/g1/vision.py
```

核心类：

- `G1CameraConfig`: 配置 `192.168.123.164:5555`、当前 G1 DDS 网卡、`ego_view` 和 `ego_view_depth_m`。
- `G1CameraSample`: 一帧 RGB、depth image、可选 metric depth。
- `G1CameraApi`: 已注册为 `G1CameraApi`，可把相机帧转成 `G1RealLowLevel` 接收的 `camera_top` observation。

调试抓图：

```bash
.venv/bin/python tools/g1_vision_bridge/capture_zmq_once.py --output-dir ./output
```

相机到 SAM3 的 smoke test，默认检测目标是 `displayer`：

```bash
.venv/bin/python tools/g1_vision_bridge/smoke_test_sam3_displayer.py \
  --start-sam3-server \
  --prompt displayer \
  --output-dir ./output/g1_sam3_smoke
```

如果已经通过 CaP-X 配置或其他终端启动了 SAM3 `127.0.0.1:8114`，可以省略
`--start-sam3-server`。

可视化：

```bash
.venv/bin/python tools/g1_vision_bridge/viewer_zmq.py
```

转发给 CaP-X 的 `G1RealLowLevel`：

```bash
.venv/bin/python tools/g1_vision_bridge/client_zmq_to_capx.py \
  --intrinsics-yaml env_configs/g1/g1_d435_depth_640x480_intrinsics.yaml \
  --extrinsics-yaml env_configs/g1/g1_d435_depth_optical_extrinsics_torso.yaml
```

注意：当前 server 如果发布 `ego_view_depth_m`，API 会把它作为米制 `float32 depth_m` 接入
CaP-X 抓取流程；如果只发布旧的 `ego_view_depth` JPEG，API 只把它当作可视化深度图，不会假装成
真实深度。SAM3 分割可以只看 RGB，但 `get_object_pose()`、`sample_grasp_center_pose()` 和 `sample_grasp_pose()` 的 3D 抓取流程需要真实
metric depth、相机内参和外参。

Dry-run 到 IK 后停止，并保存 SAM3、点云、物体/抓取位姿和 HTML report：

```bash
CAPX_G1_DRY_RUN=true \
.venv/bin/python tools/g1_vision_bridge/g1_grasp_bottle_preflight.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --object-name "plastic water bottle" \
  --output-dir ./outputs/g1_grasp_bottle_preflight \
  --observation-port 9000 \
  --observation-timeout-s 60 \
  --sam3-url http://127.0.0.1:8114 \
  --graspnet-url http://127.0.0.1:8115 \
  --pyroki-url http://127.0.0.1:8116 \
  --z-approach 0.10 \
  --keep-started-api-servers
```

输出入口是 `./outputs/g1_grasp_bottle_preflight/preflight_report.html` 和
`./outputs/g1_grasp_bottle_preflight/preflight_summary.json`。这个 preflight 不会下发关节命令，只会
求解 PyRoKi IK 并保存最终右臂 7-DoF 关节角。


## Manual+SLAM 抓瓶搬运持久 Demo runner

`tools/g1_bottle_carry_demo.py` 只控制右臂 `arm_action` 和右 Dex3，不发布
`pedal_command`，也不创建 `rt/lowcmd` writer。locomotion 由 Humanoid
`manual_whole_body_api --with-official-slam` 独立拥有，两进程只通过 operator 按
Enter 确认阶段，不增加 IPC。

这个持久 runner 不会自行启动 API servers；同一台机器另一个终端必须运行
`capx/serving/launch_servers.py --config-path env_configs/g1/g1_grasp_bottle.yaml`，并确认
8114/8115/8116 分别是真正的 SAM3、ContactGraspNet 和 PyRoKi。若任一端口被其它程序
占用，先释放端口再重启统一 launcher；只有端口处于 LISTEN 状态不能证明服务身份。
PyRoKi CLI 使用轻量 motion-helper namespace，避免启动 8116 时触发完整 Cap-X API/env
注册图的循环导入。

先按本页启动 RGB-D observation bridge 和 SAM3/ContactGraspNet/PyRoKi 服务。真实
运行必须显式双重确认：

```bash
cd /home/peilab/development/cap-x
export UNITREE_NETWORK_INTERFACE=<G1 DDS interface>
export CAPX_G1_DRY_RUN=false
.venv/bin/python tools/g1_bottle_carry_demo.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --enable-real
```

流程如下：

1. Humanoid manual 终端到达桌 A 并完成小步站位微调后，在 Cap-X 第一次提示按 Enter。
2. runner 复用已验证的 `sample_grasp_center_pose()`、
   `grasp_at_pinch_center(..., lift_dz=0.12)` 和右侧抬臂姿态；同一环境继续存活，
   `G1RealLowLevel` 的 arm hold worker 持续刷新最后 target。runner 同时以
   `arm_hold_status()` 监控 target、worker 线程、暂停状态和发布错误；任何一项异常都会
   锁存 `failed`，禁止进入桌 B 放置。抓取位姿只在抓取、侧抬和首次持臂健康检查全部
   成功后保存，部分失败不能复用该位姿。
3. 在 Humanoid manual 终端导航并微调到桌 B；确认瓶子和机器人稳定后，在 Cap-X
   第二次提示按 Enter。
4. runner 复用桌 A 保存的 torso-frame pose，依次执行 hover、下降、开手、后上方
   撤回和右侧抬臂。两桌默认同高；Demo 结果由现场观察与录像确认。

软件 dry-run 仅把低层输出强制为 dry-run；仍需要 observation 与视觉/IK 服务：

```bash
.venv/bin/python tools/g1_bottle_carry_demo.py --dry-run
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_g1_bottle_carry_demo.py tests/test_unitree_g1_task_config.py \
  tests/integrations/test_g1_real_unitree_adapter.py -q
```

不要在持瓶导航期间退出、reset 或重新实例化 Cap-X 环境，否则 arm hold 和保存的桌 A
抓取位姿都会丢失。

## 左臂控制

`G1LeftRealControlApi` 为 coding agent 提供与 `G1RealControlApi` 完全相同的函数集合：`goto_pose`、`move_to_joints`、`sample_grasp_center_pose`、`grasp_at_pinch_center`、`move_to_pregrasp_side_pose`、`open_gripper`、`close_gripper`、`close_index_pinch`、`close_middle_pinch`、`set_gripper_trigger_squeeze` 和 `move_hand_joints`。

使用 `env_configs/g1/g1_grasp_bottle_left.yaml` 时，低层环境、Gateway `arm_action[0:7]`、PyRoKi `left_hand_palm_link` 和 Dex3 left topic 会一起切换到左侧。左臂 7 个关节顺序为 `left_shoulder_pitch_joint`、`left_shoulder_roll_joint`、`left_shoulder_yaw_joint`、`left_elbow_joint`、`left_wrist_roll_joint`、`left_wrist_pitch_joint`、`left_wrist_yaw_joint`。

左手闭合方向按 URDF 镜像处理；首次真机使用左侧预备姿态前，先以 `CAPX_G1_DRY_RUN=true` 检查轨迹，再由现场人员在站立平衡状态下执行。
