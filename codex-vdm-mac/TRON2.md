# 从 Mac 启动 TRON2 评测

在 Mac 终端使用 Python 3.12 或更新版本。默认用 SSH 别名 `4090w` 连接 Linux 仿真主机，`A800` 连接推理主机。可用 `--sim-host` 和 `--a800-host` 指定其他 SSH 别名。M3 使用已登录 ChatGPT 的 Mac Codex CLI，可通过 `CODEX_BIN` 指定可执行文件。

在仿真主机的同一工作目录下准备 `tron2-sim/`、`tron2_manipulation_controller/` 和 `tron2-route2-runtime/`。`tron2-sim/` 是本仓库的检出目录，包含有 TRON2 依赖的 `.venv`。`tron2-route2-runtime/` 存放 `sdk/` 和 `opt/`。资产及运行环境见 [TRON2 开发文档](../docs/tron2.md)。在新批次使用 `--sim-base` 或 `CAPX_TRON2_SIM_BASE` 指定该远端工作目录的绝对路径。

从本仓库根目录运行。若已有 Qwen3.8-27B 服务，用 `--a800-port` 指定它在推理主机上的回环端口。

```bash
python3 codex-vdm-mac/run_tron2_mac.py --run-id tron2_m2_01 \
  --sim-base /path/to/simulation/workspace --a800-port 18089 \
  --conditions v2:M2 --seeds 1 2 3 4 5 --max-code-blocks 8
```

看板地址为 [http://127.0.0.1:8120/](http://127.0.0.1:8120/)。选择批次里的 episode 即可查看请求、回复、当前相机画面和动作反馈。结束后同一页面提供合并录像。启动器启动看板后，页面可以独立打开或关闭。运行期间 Mac 保持唤醒和 SSH 连通。

`Ctrl+C` 停止当前试次。使用原 run ID 继续。

```bash
python3 codex-vdm-mac/run_tron2_mac.py --resume tron2_m2_01 --a800-port 18089
```

续跑沿用 manifest 中的档位、种子、代码预算和仿真路径。结果和录像齐全的 episode 会跳过。未完成的 episode 从重置开始新 attempt，旧 attempt 的产物保存在同一目录。批次记录、汇总和录像位于仓库的 `outputs/tron2_dashboard_runs/<run-id>/`。

省略 `--a800-port` 时，启动器选一张空闲的 A800 并启动带视觉的 Qwen3.8-27B Graph + MTP 服务。可用 `--a800-base` 或 `CAPX_A800_BASE` 指定推理主机上的绝对路径。续跑会默认采用 manifest 中保存的值，未设置时使用远端 home。该路径下需要 `pear_official_runtime/` 和 `models/Qwen3.8-27B/`。可用 `--sim-gpu` 和 `--a800-gpu` 指定卡，或用 `--dashboard-port` 指定看板端口。复用的服务由原启动者管理。
