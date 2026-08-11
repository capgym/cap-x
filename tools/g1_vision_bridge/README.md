# G1 RGB-D 视觉 Bridge

这个目录是独立的 G1 视觉信息桥接工具，用于把 G1 机载电脑上的 RGB-D
相机数据送到本机 CaP-X。

默认 RGB-D bridge 数据流：

```text
G1 机载电脑 RGB-D 相机
  -> publisher_realsense.py
  -> TCP msgpack
  -> 本机 client_to_capx.py
  -> CaP-X G1RealLowLevel msgpack server, 默认 127.0.0.1:9000
```

`publisher_realsense.py` 默认调用 librealsense/`pyrealsense2` 的设备出厂标定，
把 RGB 重采样到 depth optical frame，并直接读取当前 depth stream 的内参 K。
因此不需要手工填写 RGB↔depth 外参；传入的安装外参仍然是 `T_torso_depth`。

如果机器人端已经启动了方案 A 的 camera server，可以直接使用本目录的 ZMQ 客户端：

```text
G1 机载电脑 camera server
  -> ZMQ SUB tcp://192.168.123.164:5555
  -> msgpack timestamps + images
  -> viewer_zmq.py 或 client_zmq_to_capx.py
```

CaP-X 抓取链路需要的不只是 RGB，还需要：

- `rgb`: 目标检测、分割和可视化。
- `depth_m`: 深度图，单位为米，用于生成点云和 3D 抓取候选。
- `intrinsics`: 3x3 相机内参矩阵。
- `T_world_camera`: 4x4 相机外参，必须和 PyRoKi/G1 控制使用同一个 world/base 坐标系。

## 文件说明

- `protocol.py`: length-prefixed msgpack 协议、numpy 序列化、CaP-X observation 转换。
- `calibration.py`: 读取相机外参 YAML。
- `publisher_realsense.py`: 在 G1 机载电脑上采集 RealSense RGB-D 并发布。
- `client_to_capx.py`: 在本机接收 RGB-D 帧，并转发到 CaP-X 的 `G1RealLowLevel`。
- `viewer.py`: 在本机可视化 publisher 发来的 RGB 和 depth。
- `zmq_camera_client.py`: 方案 A 的 ZMQ/msgpack/JPEG 解码层。
- `viewer_zmq.py`: 在本机可视化机器人端 ZMQ camera server 发来的 RGB。
- `client_zmq_to_capx.py`: 把机器人端 ZMQ camera server 的图像转成 CaP-X observation。
- `capture_zmq_once.py`: 抓取一帧 RGB/depth image 到本机目录。
- `extrinsics_example.yaml`: 外参 YAML 示例，不能直接用于真实抓取。

核心 API 已按 Franka integration 的方式包装在 `capx/integrations/g1/vision.py`：

- `G1CameraConfig`: 配置 robot IP、端口、网口、图像 key、内参、外参和测试深度。
- `G1CameraSample`: 一帧解码后的 RGB、可视化 depth image、可选 metric depth。
- `G1CameraApi(ApiBase)`: 负责 ZMQ 收包、msgpack/JPEG 解码、保存图像、生成 CaP-X observation。

`G1CameraApi` 已注册为 CaP-X API 名称 `G1CameraApi`，和 `G1RealControlApi` 一样走
`capx.integrations.register_api(...)`。

对应 CaP-X 流程：

```text
G1 camera server
  -> G1CameraApi.capture_camera_sample()
  -> G1CameraSample
  -> G1CameraApi.sample_to_capx_observation()
  -> {"camera_top": ..., "timestamp": ...}
  -> G1RealLowLevel observation_server:9000
  -> G1RealControlApi.get_object_pose/sample_grasp_pose
  -> SAM3 + depth point cloud + grasp planning
  -> G1RealControlApi.goto_pose
```

## G1 机载电脑安装

建议在机载电脑上创建独立 Python 3.10 环境。RealSense 需要 `pyrealsense2`。

```bash
cd /path/to/cap-x/tools/g1_vision_bridge
python -m pip install -r requirements-publisher-realsense.txt
```

如果机载电脑是 Jetson/ARM，`pyrealsense2` 可能需要按 Intel RealSense 官方方式从源码安装。

## 本机安装

本机使用当前 CaP-X `.venv` 即可，相关依赖已经在 CaP-X 环境中可用：

```bash
cd /home/peilab/development/cap-x
.venv/bin/python -c "from tools.g1_vision_bridge.zmq_camera_client import decode_zmq_camera_payload; print('ok')"
```

如果单独部署 client，也可以安装最小依赖：

```bash
python -m pip install -r tools/g1_vision_bridge/requirements-client.txt
```

## 方案 A：连接已有 ZMQ camera server

你的当前机器人端配置：

- robot IP: `192.168.123.164`
- ZMQ SUB endpoint: `tcp://192.168.123.164:5555`
- 本机网口: `enx6c1ff7c1192d`
- payload: `msgpack`，包含 `timestamps` 和 `images`
- image: raw JPEG bytes 或 base64 JPEG string
- 当前默认 RGB key: `ego_view`
- 当前默认 metric depth key: `ego_view_depth_m`

API 也兼容旧的 `ego_view_depth` JPEG depth image，但这种 JPEG 只用于显示，不能作为真实抓取的
米制深度。

注意：旧 5555 ZMQ payload 本身没有携带 RealSense 出厂内参或 RGB-D 对齐元数据。
即使 RGB 与 depth 分辨率相同，也不能据此认定已经对齐。真实抓取只应在机载 publisher
已用官方 SDK 对齐到 depth frame，且 CaP-X 使用同一 depth K/`T_torso_depth` 时走此路径；
否则改用下面的 `publisher_realsense.py` 官方标定路径。

先确认本机路由确实从指定网口出：

```bash
ip route get 192.168.123.164
ping -I enx6c1ff7c1192d 192.168.123.164
```

如果 `ip route get` 显示的出接口不是 `enx6c1ff7c1192d`，需要先调整系统路由。
代码会尝试对 ZMQ socket 设置 `BINDTODEVICE=enx6c1ff7c1192d`，但这个选项取决于
Linux/libzmq 支持和当前用户权限；最终出接口仍以系统路由为准。

只看机器人端发来的 RGB：

```bash
cd /home/peilab/development/cap-x
.venv/bin/python tools/g1_vision_bridge/viewer_zmq.py
```

等价显式写法：

```bash
.venv/bin/python tools/g1_vision_bridge/viewer_zmq.py \
  --robot-host 192.168.123.164 \
  --robot-port 5555 \
  --interface enx6c1ff7c1192d
```

如果 server 的 `images` 里有多个 key，用 `--camera-key` 指定其中一个：

```bash
.venv/bin/python tools/g1_vision_bridge/viewer_zmq.py \
  --camera-key head
```

无图形界面时只打印收到的数据：

```bash
.venv/bin/python tools/g1_vision_bridge/viewer_zmq.py --no-window
```

抓取一帧 RGB 和 depth image 到 `./output`：

```bash
.venv/bin/python tools/g1_vision_bridge/capture_zmq_once.py \
  --output-dir ./output
```

抓取一帧并交给 SAM3 检测 `displayer`：

```bash
.venv/bin/python tools/g1_vision_bridge/smoke_test_sam3_displayer.py \
  --start-sam3-server \
  --prompt displayer \
  --output-dir ./output/g1_sam3_smoke
```

输出包括 RGB、`depth_m.npy`、SAM3 mask/box 可视化图和
`sam3_displayer_summary.json`。如果 `127.0.0.1:8114` 的 SAM3 服务已经启动，可以省略
`--start-sam3-server`。

把 ZMQ 图像转发给 CaP-X：

```bash
.venv/bin/python tools/g1_vision_bridge/client_zmq_to_capx.py \
  --robot-host 192.168.123.164 \
  --robot-port 5555 \
  --interface enx6c1ff7c1192d \
  --capx-host 127.0.0.1 \
  --capx-port 9000 \
  --intrinsics-yaml env_configs/g1/g1_d435_depth_640x480_intrinsics.yaml \
  --extrinsics-yaml env_configs/g1/g1_d435_depth_optical_extrinsics_torso.yaml
```

如果当前 ZMQ server 只发布 RGB JPEG，不发布 metric depth，上面的 CaP-X 转发不能用于真实抓取。
可以先用下面命令只验证链路，但它填充的是假的常量深度：

```bash
.venv/bin/python tools/g1_vision_bridge/client_zmq_to_capx.py \
  --dry-run \
  --allow-default-intrinsics \
  --allow-identity-extrinsics \
  --constant-depth-m 1.0
```

真实抓取时必须满足以下任一条件：

- 让 ZMQ camera server 在 msgpack 里同时发布 `depth_m` 或 `depth`，单位是米，shape 和 RGB 一致。
- 改用本目录的 `publisher_realsense.py` RGB-D publisher。
- 在机器人端另起 RGB-D 发布程序，把深度、内参、外参一起发给本机。

内参 YAML 可以是任一格式：

```yaml
K:
  - [fx, 0.0, cx]
  - [0.0, fy, cy]
  - [0.0, 0.0, 1.0]
```

或：

```yaml
fx: 430.0
fy: 430.0
cx: 320.0
cy: 240.0
```

## 外参配置

复制示例文件并填入真实标定结果：

```bash
cp tools/g1_vision_bridge/extrinsics_example.yaml /path/to/g1_camera_extrinsics.yaml
```

格式：

```yaml
T_world_camera:
  - [r11, r12, r13, tx]
  - [r21, r22, r23, ty]
  - [r31, r32, r33, tz]
  - [0.0, 0.0, 0.0, 1.0]
```

这里的 `world` 必须是 CaP-X/PyRoKi `goto_pose` 使用的坐标系。外参错误会直接导致
抓取位姿错误。

## Mock 自测

不用相机也可以先测试网络链路。

G1 端或任意一台机器上启动 mock publisher：

```bash
python tools/g1_vision_bridge/publisher_realsense.py \
  --source mock \
  --bind-host 0.0.0.0 \
  --port 9100 \
  --allow-identity-extrinsics
```

本机启动 CaP-X G1 env 时，`G1RealLowLevel` 会监听 `9000`。如果只想测试 client
连接逻辑，需要先运行包含 `observation_server_port: 9000` 的 G1 配置。

本机启动 bridge client：

```bash
.venv/bin/python tools/g1_vision_bridge/client_to_capx.py \
  --robot-host 192.168.5.87 \
  --robot-port 9100 \
  --capx-host 127.0.0.1 \
  --capx-port 9000
```

看到 `forwarded ... frames` 表示视觉帧已经转发到 CaP-X。

本机可视化收到的 RGB-D：

```bash
.venv/bin/python tools/g1_vision_bridge/viewer.py \
  --robot-host <publisher-ip> \
  --robot-port 9100 \
  --max-depth-m 3.0
```

viewer 窗口左侧是 RGB，右侧是 depth colormap。按 `s` 保存当前帧，按 `q`
或 `Esc` 退出。如果机器没有图形界面，可以只打印数据统计：

```bash
.venv/bin/python tools/g1_vision_bridge/viewer.py \
  --robot-host <publisher-ip> \
  --robot-port 9100 \
  --no-window
```

## 真机运行

1. 启动 CaP-X G1 配置，让 `G1RealLowLevel` 建立 `9000` observation server：

```bash
export UNITREE_NETWORK_INTERFACE=<G1 DDS 网卡>
export CAPX_G1_DRY_RUN=false

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/g1/g1_real_upper_body.yaml \
  --num-workers 1 \
  --server-url http://127.0.0.1:8110/chat/completions \
  --model <model-name> \
  --api-key <optional-api-key>
```

2. 在 G1 机载电脑启动 RealSense publisher：

```bash
python tools/g1_vision_bridge/publisher_realsense.py \
  --source realsense \
  --bind-host 0.0.0.0 \
  --port 9100 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --extrinsics-yaml env_configs/g1/g1_d435_depth_optical_extrinsics_torso.yaml
```

启动日志必须出现 `RealSense factory alignment: RGB -> depth optical frame`。
每帧携带的 K 来自相机本机的 active depth profile，不再由本机静态内参 YAML 猜测。

如果有多台 RealSense，指定序列号：

```bash
python tools/g1_vision_bridge/publisher_realsense.py \
  --source realsense \
  --serial <realsense-serial> \
  --extrinsics-yaml env_configs/g1/g1_d435_depth_optical_extrinsics_torso.yaml
```

3. 在本机启动 bridge client：

```bash
.venv/bin/python tools/g1_vision_bridge/client_to_capx.py \
  --robot-host <g1-onboard-ip> \
  --robot-port 9100 \
  --capx-host 127.0.0.1 \
  --capx-port 9000
```

4. 可选：另开一个本机终端启动 viewer，检查实时 RGB-D 数据：

```bash
.venv/bin/python tools/g1_vision_bridge/viewer.py \
  --robot-host <g1-onboard-ip> \
  --robot-port 9100
```

## 发送给 CaP-X 的 observation 格式

client 会把收到的 frame 转成 `G1RealLowLevel` 已支持的 `camera_top` 输入格式：

```python
{
    "camera_top": {
        "images": {
            "rgb": rgb_uint8_hwc,
            "depth": depth_float32_h_w_1,
        },
        "intrinsics_matrix": K_3x3,
        "pose": [x, y, z, qw, qx, qy, qz],
        "pose_mat": T_world_camera_4x4,
    },
    "timestamp": timestamp,
}
```

`G1RealLowLevel` 会把它规范化成 CaP-X visual API 使用的
`robot0_robotview.images.rgb/depth`、`intrinsics`、`pose` 和 `pose_mat`。

## 注意事项

- `--allow-identity-extrinsics` 只能用于 bench/mock 测试，不能用于真实抓取。
- depth 单位必须是米。
- RGB 是 `uint8`、`H x W x 3`、RGB 顺序；RealSense publisher 已从 BGR 转成 RGB。
- 如果 CaP-X 没有启动或 `9000` 端口不可用，client 会等待并自动重连。
- 当前 bridge 是单客户端 TCP 流；需要多客户端预览时可以另开 teleimager 做 RGB 预览。
