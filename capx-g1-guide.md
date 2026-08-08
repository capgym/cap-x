# CaP-X × Unitree G1：项目实现、塑料水瓶抓取流程与复现指南

本文按当前仓库代码说明 CaP-X 的整体能力、G1 真机接入，以及右手 Dex3 抓取塑料水瓶的完整执行链路。所有命令默认在仓库根目录执行：

~~~bash
cd /home/peilab/development/cap-x
~~~

> [!CAUTION]
> 这是真机控制流程。默认保持 **CAPX_G1_DRY_RUN=true**，只有视觉、标定、IK、Gateway、Dex3 和急停检查全部通过后才切换为 false。协同站立模式下只能由 WholeBodyGateway 发布 rt/lowcmd，不要同时运行任何直发上肢或全身 lowcmd 的程序。

> [!IMPORTANT]
> 现场失败样例 outputs/g1_grasp_bottle_fail/grasp_fail/plastic_water_bottle_grasp_debug.html 表明：G1 相机服务发送的 RGB 与深度虽然都是 640×480，但不在同一像素坐标系，SAM3 的 RGB mask 因而截取了错误的深度点云。仓库已经实现 RGB→depth 重投影，但尚未提交这台 D435 的 RGB 内参和 T_rgb_depth 标定文件。完成标定或让上游直接发布对齐帧之前，不应执行真机抓取。

## 1. 项目实现了什么

CaP-X 是一个以 Code-as-Policy 为核心的机器人操作框架：模型输出 Python 代码，代码再组合视觉、抓取规划、逆运动学和机器人控制 API 完成任务。仓库主要包含：

- **CaP-Gym**：统一的 Gymnasium 任务/环境接口，覆盖 Robosuite、LIBERO-PRO、BEHAVIOR 和真机环境。
- **CaP-Bench**：按 API 抽象层级、单轮/多轮、视觉 grounding 方式评测 coding agent。
- **CaP-Agent0**：视觉差分、多轮修正、技能库和并行集成推理。
- **CaP-RL**：通过环境奖励和 GRPO/VeRL 训练生成机器人代码的模型。
- 感知与规划服务：SAM3 文本分割、ContactGraspNet 抓取候选、PyRoKi IK，以及 OWL-ViT、SAM2、cuRobo 等可选组件。
- 多种机器人后端：Franka、Unitree G1、R1Pro 等；G1 路径支持左右 7-DoF 手臂和左右 Dex3 手。

仓库职责划分：

| 路径 | 职责 |
| --- | --- |
| capx/envs/launch.py | 加载 YAML、启动 API 服务、进入单次/批量 trial |
| capx/envs/tasks/ | 任务 prompt、oracle code 和 code-execution 环境 |
| capx/envs/simulators/g1_real.py | G1 观测、手臂轨迹、Dex3 命令和 dry-run |
| capx/integrations/g1/ | G1 控制 API、Gateway、SDK、抓取几何和相机适配 |
| capx/serving/ | SAM3、ContactGraspNet、PyRoKi 等本地服务 |
| env_configs/g1/ | 左右手任务、URDF、相机内外参与服务配置 |
| tools/g1_vision_bridge/ | 机载 ZMQ RGB-D 到 CaP-X 的桥接、viewer 和 preflight |
| tools/g1_dex3_*.py | Dex3 只读诊断与开合循环测试 |
| tests/ | 配置、序列化、左右手路由、视觉桥和 preflight 测试 |

## 2. G1 实现边界与关键文件

右手塑料水瓶任务由以下文件组成：

- 任务与固定 oracle：capx/envs/tasks/unitree_g1/grasp_bottle.py
- 真机配置：env_configs/g1/g1_grasp_bottle.yaml
- Agent 可调用 API：capx/integrations/g1/control.py
- Dex3 闭合几何与前向抓取策略：capx/integrations/g1/grasp.py
- Gateway LCM 桥：capx/integrations/g1/gateway.py
- Unitree 手臂/Dex3 DDS 桥：capx/integrations/g1/sdk.py
- 真机低层环境：capx/envs/simulators/g1_real.py
- RGB-D 接口与对齐：capx/integrations/g1/vision.py
- G1 29-DoF + 双 Dex3 URDF：env_configs/g1/g1_29dof_with_hand.urdf

左手使用 grasp_bottle_left.py 和 g1_grasp_bottle_left.yaml；API、Gateway arm slice、PyRoKi target link、预备姿态和 Dex3 topic 会一起切到左侧。

### 2.1 端口、通道与 topic

| 地址/通道 | 生产者 → 消费者 | 内容 |
| --- | --- | --- |
| tcp://192.168.123.164:5555 | G1 相机服务 → vision bridge | ZMQ/msgpack/JPEG RGB 与 metric depth |
| 127.0.0.1:9000 | vision bridge → G1RealLowLevel | 统一 camera_top observation |
| 127.0.0.1:9010 | vision bridge → 浏览器 | RGB、depth、SAM3 和抓取点 viewer |
| 127.0.0.1:8114 | G1RealControlApi → SAM3 | 文本提示分割 |
| 127.0.0.1:8115 | G1RealControlApi → ContactGraspNet | 深度、内参、segmap 到抓取候选 |
| 127.0.0.1:8116 | G1RealControlApi → PyRoKi | 掌心目标到关节 IK |
| LCM arm_action | CaP-X → WholeBodyGateway | 左右臂共 14 个关节目标 |
| LCM body_control_data | Gateway → CaP-X | 29 关节状态；用于保留未控制手臂 |
| LCM state_estimator_data | Gateway → CaP-X | 机体姿态 |
| DDS rt/dex3/right/cmd | CaP-X → 右 Dex3 | 7 电机手指命令 |
| DDS rt/lf/dex3/right/state | 右 Dex3 → CaP-X | 右手状态 |
| DDS rt/dex3/left/cmd | CaP-X → 左 Dex3 | 左手命令 |
| DDS rt/dex3/left/state | 左 Dex3 → CaP-X | 左手状态 |

Gateway 的 14-DoF arm_action 中 [0:7] 是左臂、[7:14] 是右臂。CaP-X 只替换选中侧，另一侧从最新 body_control_data.q[15:29] 保留。

## 3. 塑料水瓶抓取的完整执行流程

~~~text
任务 YAML / oracle 或 LLM 生成代码
              │
              ▼
G1Camera ZMQ ──► RGB-D bridge ──► :9000 G1RealLowLevel
                       │
                       ├─ RGB/depth 像素对齐
                       ├─ depth K
                       └─ T_torso_camera
              │
              ▼
SAM3(:8114) ──► bottle mask
              │
              ▼
mask + metric depth + K ──► object point cloud
              │
              ▼
ContactGraspNet(:8115) ──► candidate/contact points
              │
              ▼
G1 front policy ──► closed-pinch center target
              │
              ▼
Dex3 URDF offset ──► hand_palm_link target
              │
              ▼
PyRoKi(:8116) ──► 7-DoF arm joints
              │
       ┌──────┴─────────┐
       ▼                ▼
Gateway arm_action   Dex3 DDS
       │                │
       └──────► 抓取并抬升
~~~

### 3.1 启动与代码执行

capx/envs/launch.py 加载 g1_grasp_bottle.yaml，自动启动：

- SAM3：CUDA，端口 8114，权重 capx/model_weights/sam3/sam3.pt。
- ContactGraspNet：端口 8115。
- G1 专用 PyRoKi：端口 8116，root_link=torso_link，target_link=right_hand_palm_link。

随后构造 UnitreeG1GraspBottleCodeEnv、G1RealLowLevel、G1RealControlApi 和 G1CameraApi。--use-oracle-code True 直接运行仓库内固定代码；否则通过 OpenAI-compatible 接口请求模型生成 Python，再由 CodeExecutionEnvBase 执行。

当前 oracle 等价于：

~~~python
grasp_pos, grasp_quat = sample_grasp_center_pose("plastic water bottle")
grasp_at_pinch_center(
    grasp_pos,
    grasp_quat,
    trigger=1.0,
    squeeze=1.0,
    lift_dz=0.12,
)
move_to_pregrasp_side_pose()
~~~

真机第一次复现应使用 oracle，排除模型随机生成动作这一变量。

### 3.2 RGB-D 接入与坐标系

vision bridge 从 G1 相机 server 读取：

- ego_view：RGB；
- ego_view_depth_m：米制 float32 深度；
- depth 内参；
- T_world_camera，当前 world 是 torso_link。

数据被转换为 camera_top，再由 G1RealLowLevel 规范化成 robot0_robotview。真实 3D 抓取必须同时满足：

1. RGB 和深度逐像素对齐；
2. depth 单位是米，而不是彩色深度 JPEG；
3. 内参对应最终 depth 像素坐标系；
4. T_world_camera 把 depth optical frame 正确变换到 torso_link。

### 3.3 SAM3、点云与 ContactGraspNet

sample_grasp_center_pose() 先复用共享的 sample_grasp_pose()：

1. 对 RGB 运行文本提示 plastic water bottle；
2. 选择 SAM3 分数最高的 mask；
3. 用 mask、米制 depth 和相机内参反投影点云；
4. 把 depth、内参、segmap 发送给 ContactGraspNet；
5. 保存候选、分数、contact points 和 3D HTML debug report。

ContactGraspNet 默认只接受约 0.2–2.0 m 的有效深度。RGB 中能看到瓶子并不代表抓取输入正确；mask 必须覆盖瓶子对应的有效深度。

### 3.4 G1 front policy 与 Dex3 几何

共享 sample_grasp_pose() 会保留网络旋转并添加旧的局部 +Z 0.12 m 偏移。G1 推荐 API 会改写结果：

- 移除旧的 +Z 0.12 m 偏移；
- target XY 优先取最佳 ContactGraspNet contact points 的质心；
- target Z 取 SAM3 物体点云包围盒的上下中心；
- 丢弃网络输出旋转，固定使用 quaternion_wxyz=[1, 0, 0, 0]。

这个 target 是 Dex3 **闭合后拇指/食指/中指指尖中心**，不是 hand_palm_link。closed_pinch_center_offset() 从 29-DoF URDF 和闭合手指角计算指尖中心，palm_pose_from_pinch_center_pose() 再把 pinch-center target 反算为掌心 IK target。

### 3.5 IK、预抓取、闭合和抬升

右手标准动作顺序：

1. 首次 Cartesian 动作前，将右臂移到侧抬预备位 [0, -1.25, 0, 1.1, 0, 0, 0]。
2. 可选地先按目标 trigger/squeeze 合手，移动到目标前方。
3. pinch center 沿 world -X 后退 0.20 m，形成预抓取点。
4. 张手：trigger=0, squeeze=0。
5. 固定 Z，规划 50 个 Cartesian waypoint；每点经 PyRoKi 求解为 7-DoF 关节角。
6. 默认以 20 ms 周期连续发送，并额外保持最终点 0.4 s。
7. 三指闭合：trigger=1, squeeze=1。trigger 控制拇指+食指，squeeze 控制拇指+中指。
8. pinch center 沿 world +Z 抬升 0.12 m。
9. 最后再次移到右侧抬升预备位，使 Gateway 从侧方目标进入超时阻尼。

阻塞关节动作默认还会做 50 步关节插值。Gateway 接收选中手臂目标并维持下肢 Homie 控制；Dex3 手指不经过 Gateway，而是由 CaP-X 直接发往对应 DDS hand topic。

## 4. 复现前准备

### 4.1 硬件与外部服务

- Unitree G1，目标侧安装 Dex3 手；
- D435/D435i RGB-D，相机 server 已在 192.168.123.164:5555 发布；
- 本机与 G1 DDS/相机网段连通；
- WholeBodyGateway 已编译；
- OpenHomie 站立/平衡策略和生成的 LCM Python types 可用；
- 现场操作员全程看护急停，机械臂工作空间清空，使用轻质塑料瓶。

当前开发机已存在以下资产，换机器时必须重新确认：

~~~text
capx/model_weights/sam3/sam3.pt
capx/third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints
capx/third_party/unitree_sdk2_python
/home/peilab/development/WholeBodyGateway/robot_control_gateway/build-g1-v2/bin/humanoid_g1_control_gateway
/home/peilab/development/OpenHomie/HomieDeploy/g1_gym_deploy/lcm_types
~~~

### 4.2 Python 环境

~~~bash
uv python install 3.10
uv venv -p 3.10
uv sync --extra contactgraspnet

uv --cache-dir /home/peilab/development/cap-x/.uv-cache \
  pip install -p .venv/bin/python \
  -e capx/third_party/unitree_sdk2_python

.venv/bin/python -c "import lcm, unitree_sdk2py, zmq, cv2; print('G1 runtime imports OK')"
~~~

LCM 安装取决于 WholeBodyGateway/OpenHomie 的构建环境，仓库 pyproject.toml 没有声明它；若导入失败，应复用外部控制栈对应的 LCM 版本。

### 4.3 环境变量与网络

当前机器的 G1 网卡是 enx00e04c3601b7，不要依赖代码中的旧默认值 enx6c1ff7c1192d：

~~~bash
export G1_NET_IF=enx00e04c3601b7
export UNITREE_NETWORK_INTERFACE="$G1_NET_IF"
export CAPX_G1_GATEWAY_LCM_TYPES_DIR=/home/peilab/development/OpenHomie/HomieDeploy/g1_gym_deploy/lcm_types
export NO_PROXY=127.0.0.1,localhost,192.168.123.164
export no_proxy="$NO_PROXY"

ip -br link show "$G1_NET_IF"
ip route get 192.168.123.164
ping -c 2 192.168.123.164
~~~

## 5. RGB/深度对齐：真机执行前的硬门槛

仓库现有标定文件只有：

- g1_d435_depth_640x480_intrinsics.yaml：depth K；
- g1_d435_depth_optical_extrinsics_torso.yaml：T_torso_depth_optical。

如果相机 server 已把 RGB 重投影到 depth frame，直接发布对齐后的 ego_view 即可。否则需补充：

~~~yaml
# env_configs/g1/g1_d435_rgb_640x480_intrinsics.yaml
fx: REPLACE_WITH_CALIBRATED_FX
fy: REPLACE_WITH_CALIBRATED_FY
cx: REPLACE_WITH_CALIBRATED_CX
cy: REPLACE_WITH_CALIBRATED_CY
distortion_coeffs: [K1, K2, P1, P2, K3]
~~~

~~~yaml
# env_configs/g1/g1_d435_depth_to_rgb_extrinsics.yaml
# T_rgb_depth: depth optical frame -> RGB/color optical frame
T_rgb_depth:
  - [R00, R01, R02, TX]
  - [R10, R11, R12, TY]
  - [R20, R21, R22, TZ]
  - [0.0, 0.0, 0.0, 1.0]
~~~

不要把占位值写入真实文件；数据应来自该相机的 factory calibration 或可靠现场标定。

先单独启动 SAM3：

~~~bash
uv run --no-sync --active python -m capx.serving.launch_sam3_server \
  --host 127.0.0.1 \
  --port 8114 \
  --device cuda \
  --checkpoint-path capx/model_weights/sam3/sam3.pt
~~~

另一个终端仅做相机解码、对齐和 viewer：

~~~bash
uv run --no-sync --active python tools/g1_vision_bridge/client_zmq_to_capx.py \
  --robot-host 192.168.123.164 \
  --robot-port 5555 \
  --interface "$G1_NET_IF" \
  --intrinsics-yaml env_configs/g1/g1_d435_depth_640x480_intrinsics.yaml \
  --depth-intrinsics-yaml env_configs/g1/g1_d435_depth_640x480_intrinsics.yaml \
  --rgb-intrinsics-yaml env_configs/g1/g1_d435_rgb_640x480_intrinsics.yaml \
  --depth-to-rgb-extrinsics-yaml env_configs/g1/g1_d435_depth_to_rgb_extrinsics.yaml \
  --align-rgb-to-depth \
  --extrinsics-yaml env_configs/g1/g1_d435_depth_optical_extrinsics_torso.yaml \
  --dry-run \
  --viewer \
  --viewer-port 9010 \
  --viewer-sam3 \
  --viewer-sam3-prompt "plastic water bottle" \
  --viewer-sam3-url http://127.0.0.1:8114
~~~

打开 http://127.0.0.1:9010/，按 R 重新采样。必须同时满足：

- aligned RGB 物体轮廓与 depth 边界一致；
- SAM3 mask 覆盖瓶身，不包含大面积桌面或背景；
- viewer grasp point 落在瓶身内部，world 坐标与现场相符；
- mask 内有充足米制有效深度；
- 3D 点云没有错位、镜像、翻转或落到桌面下方。

仅 rgb.shape == depth.shape == (480, 640) 不能证明完成对齐。

> [!WARNING]
> 不要为 G1 抓取长期运行通用的 launch_servers.py --profile default。该 profile 的 PyRoKi 没有 G1 参数时默认加载 Panda；如果它占用 8116，G1 YAML 会误以为正确服务已存在并跳过启动。G1 抓取应让任务 YAML 启动 torso_link → right_hand_palm_link 的专用服务。

## 6. 分阶段复现

### 阶段 A：软件回归测试

~~~bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --active pytest \
  tests/test_unitree_g1_task_config.py \
  tests/integrations/test_g1_real_unitree_adapter.py \
  tests/test_g1_vision_bridge.py \
  tests/test_g1_grasp_bottle_preflight.py \
  tests/test_g1_dex3_cycle_script.py \
  tests/test_g1_dex3_diagnostic_script.py \
  -q
~~~

### 阶段 B：Dex3 状态与小范围测试

先做只读诊断，不发布手指命令：

~~~bash
.venv/bin/python tools/g1_dex3_diagnostic.py \
  --side right \
  --interface "$G1_NET_IF" \
  --duration-s 5 \
  --report-json outputs/g1_dex3_right_diagnostic.json
~~~

g1_dex3_cycle.py 默认也是 dry-run：

~~~bash
.venv/bin/python tools/g1_dex3_cycle.py \
  --interface "$G1_NET_IF" \
  --cycles 1
~~~

只有状态、温度、力矩、机械结构和工作空间均正常时，才由现场人员决定是否加 --execute。若有损坏手指，优先使用 diagnostic 的 --exclude-finger 和显式确认机制，不要运行全手 cycle。

### 阶段 C：与正式任务同链路的 dry-run

终端 C1 启动 vision bridge。若上游 RGB 已在 depth frame，删除 rgb-intrinsics、depth-to-rgb、align-rgb-to-depth 三项及额外 depth-intrinsics 项；否则使用已验证的标定：

~~~bash
uv run --no-sync --active python tools/g1_vision_bridge/client_zmq_to_capx.py \
  --robot-host 192.168.123.164 \
  --robot-port 5555 \
  --interface "$G1_NET_IF" \
  --capx-host 127.0.0.1 \
  --capx-port 9000 \
  --intrinsics-yaml env_configs/g1/g1_d435_depth_640x480_intrinsics.yaml \
  --depth-intrinsics-yaml env_configs/g1/g1_d435_depth_640x480_intrinsics.yaml \
  --rgb-intrinsics-yaml env_configs/g1/g1_d435_rgb_640x480_intrinsics.yaml \
  --depth-to-rgb-extrinsics-yaml env_configs/g1/g1_d435_depth_to_rgb_extrinsics.yaml \
  --align-rgb-to-depth \
  --extrinsics-yaml env_configs/g1/g1_d435_depth_optical_extrinsics_torso.yaml \
  --viewer \
  --viewer-port 9010 \
  --viewer-sam3 \
  --viewer-sam3-prompt "plastic water bottle" \
  --viewer-sam3-url http://127.0.0.1:8114
~~~

看到 Waiting for 127.0.0.1:9000 是正常的。终端 C2 运行 oracle dry-run：

~~~bash
export CAPX_G1_DRY_RUN=true

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --num-workers 1 \
  --total-trials 1 \
  --record-video True \
  --use-oracle-code True
~~~

该模式执行正式任务的 front-policy、Dex3 几何、PyRoKi IK 和完整目标序列，但 arm/Gateway/Dex3 bridge 只记录命令，不发送真机。

还可单独运行 perception preflight：

~~~bash
CAPX_G1_DRY_RUN=true \
.venv/bin/python tools/g1_vision_bridge/g1_grasp_bottle_preflight.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --object-name "plastic water bottle" \
  --output-dir outputs/g1_grasp_bottle_preflight \
  --observation-port 9000 \
  --observation-timeout-s 60 \
  --sam3-url http://127.0.0.1:8114 \
  --graspnet-url http://127.0.0.1:8115 \
  --pyroki-url http://127.0.0.1:8116 \
  --z-approach 0.10
~~~

preflight 会自己占用 9000，不要与 launch 同时运行。它停在 PyRoKi IK，不发送关节命令；但它审计原始 ContactGraspNet grasp pose，不完全等同于正式 oracle 的固定 front-policy pinch-center 轨迹，不能替代上一项 dry-run。

重点检查：

~~~text
outputs/g1_grasp_bottle_preflight/preflight_report.html
outputs/g1_grasp_bottle_preflight/preflight_summary.json
outputs/g1_grasp_bottle/grasp_debug/plastic_water_bottle_grasp_debug.html
outputs/oracle/g1_grasp_bottle/
~~~

### 阶段 D：四终端真机抓取

#### 终端 1：WholeBodyGateway

~~~bash
cd /home/peilab/development/cap-x

/home/peilab/development/WholeBodyGateway/robot_control_gateway/build-g1-v2/bin/humanoid_g1_control_gateway \
  "$G1_NET_IF" \
  --arm-timeout-mode damping
~~~

Gateway 是唯一允许发布 rt/lowcmd 的进程。

#### 终端 2：OpenHomie 下肢站立与平衡

~~~bash
conda activate homie310
cd /home/peilab/development/OpenHomie/HomieDeploy
python g1_gym_deploy/scripts/deploy_policy.py
~~~

按 OpenHomie 现场流程让 G1 站稳。确认 Gateway 接管下肢命令、机体无明显振荡后，才启动上肢任务。

#### 终端 3：对齐后的 vision bridge

使用阶段 C1 已验证的同一命令，不要加 --dry-run。bridge 可先启动并等待 9000；终端 4 启动后自动连接。浏览器继续打开 http://127.0.0.1:9010/。

#### 终端 4：CaP-X oracle 真机执行

~~~bash
cd /home/peilab/development/cap-x
source .venv/bin/activate

export UNITREE_NETWORK_INTERFACE="$G1_NET_IF"
export CAPX_G1_DRY_RUN=false
export NO_PROXY=127.0.0.1,localhost,192.168.123.164
export no_proxy="$NO_PROXY"

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --num-workers 1 \
  --total-trials 1 \
  --record-video True \
  --use-oracle-code True
~~~

切换为 false 前逐项口令确认：

- 急停操作员就位；
- Gateway 是唯一 rt/lowcmd publisher；
- 右臂/Dex3 与配置一致；
- viewer 的 mask、depth、点云和 world grasp point 已复核；
- 8116 使用 G1 URDF、torso_link → right_hand_palm_link；
- 瓶子在可达空间，桌面和人体不在预抓取/抬升路径中；
- dry-run 的 IK 关节角和相邻 waypoint 无突跳。

日志一旦出现 move_to_joints reached timeout、点位异常、机体失稳或手臂未跟随，应立即中止。当前实现遇到关节收敛超时只打印警告，仍可能继续后续步骤，不能忽略。

### 阶段 E：可选 LLM 生成代码

oracle 真机流程稳定后才考虑模型模式。不要把 key 写入 Markdown、YAML、shell history 或命令参数：

~~~bash
export CAPX_LLM_API_KEY="YOUR_OPENAI_COMPATIBLE_API_KEY"

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/g1/g1_grasp_bottle.yaml \
  --num-workers 1 \
  --total-trials 1 \
  --server-url https://llmapi.paratera.com/v1/chat/completions \
  --model Qwen3.6-Plus \
  --temperature 1.0 \
  --record-video True
~~~

PARATERA_API_KEY 也受支持，但 CAPX_LLM_API_KEY 优先。LLM 返回的是会直接调用真机 API 的 Python；模型模式不是安全沙箱，应先在 CAPX_G1_DRY_RUN=true 下逐次审阅生成的 code.py。

## 7. 输出与成功判定

任务会保存：

- code.py、raw_response.sh、summary.txt、all_responses.json；
- record_video=True 时的 video_turn_00.mp4 和 video_combined.mp4；
- outputs/g1_grasp_bottle/grasp_debug/*.html 的 3D 场景；
- oracle 默认输出到 outputs/oracle/g1_grasp_bottle/；
- 模型模式输出到 outputs/<model>/g1_grasp_bottle/。

当前 G1RealLowLevel.compute_reward() 固定返回 0.0，task_completed() 固定返回 False。因此即使物理抓取成功，summary 仍显示 reward_0.000_taskcompleted_0。现阶段必须由现场观察和视频人工确认：三指稳定包住瓶身、瓶底离桌、G1 保持平衡、动作后回到侧抬姿态。

## 8. 左手复现差异

~~~bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/g1/g1_grasp_bottle_left.yaml \
  --num-workers 1 \
  --total-trials 1 \
  --record-video True \
  --use-oracle-code True
~~~

差异包括：

- API：G1LeftRealControlApi；
- PyRoKi target：left_hand_palm_link；
- Gateway slice：arm_action[0:7]；
- 左臂预备位：[0, 1.25, 0, 1.1, 0, 0, 0]；
- 左 Dex3 command/state topic；
- 左手关节闭合方向按 URDF 镜像。

右手专用 g1_grasp_bottle_preflight.py 会提取 right-arm IK，不应用它验证左手最终关节角。左手必须独立完成 dry-run、Dex3 诊断和现场安全检查。

## 9. 常见故障

| 现象 | 优先检查 |
| --- | --- |
| 一直等待 observation | vision bridge、9000 监听进程、bridge 的 capx host/port |
| SAM3 能看到瓶子但抓取点在桌面/背景 | RGB-depth 是否真正对齐；不要只比较 shape |
| ContactGraspNet 无候选 | mask 内有效深度、米制单位、0.2–2.0 m 范围、瓶子点云 |
| grasp world Z 过低或点云在桌下 | T_world_camera 方向、depth optical 轴、外参是否属于当前安装 |
| PyRoKi IK 失败或姿态离谱 | 8116 的 robot/target/root、URDF、pinch-center 到 palm offset |
| body_control_data 超时 | Gateway/OpenHomie、LCM URL、LCM types、网卡和 multicast |
| Dex3 state 超时 | UNITREE_NETWORK_INTERFACE、左右手 topic、Dex3 控制器 |
| move_to_joints reached timeout | 立即中止；检查 Gateway 跟随、关节限位和目标跳变 |
| 8114/8115/8116 已占用 | 确认占用进程的模型和 PyRoKi 参数；端口可用不等于配置正确 |
| 物理成功但 summary 为失败 | G1 reward/success 尚未实现，使用视频和人工判定 |

## 10. 安全停机顺序

1. 先停止 CaP-X trial，使 arm_action 停止更新并让 Gateway arm timeout 进入 damping。
2. 观察双臂和机体稳定；必要时使用急停，不依赖软件超时。
3. 按 OpenHomie 的正常流程让机器人坐下或进入安全支撑状态。
4. 停止 vision bridge 和感知服务。
5. 机器人安全支撑后，再停止 OpenHomie 和 WholeBodyGateway。

不要在 G1 仍靠 Homie 保持站立时先杀掉唯一的 Gateway。
