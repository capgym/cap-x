# TRON2 development and evaluation

Use the TRON2 worktree for code changes. Keep LimX packages, Python environments,
model weights, and evaluation outputs in machine-local storage. Mac is the editing
and local MuJoCo inspection machine. Run the official Linux controller and the
CUDA perception stack on 4090w.

## Code layout

| Path | Responsibility |
| --- | --- |
| `capx/envs/simulators/tron2_cube_lift.py` | Single MuJoCo world, motor loop, gripper, cameras, and scoring |
| `capx/integrations/tron2/official_client.py` | Official controller's high-level WebSocket protocol |
| `capx/integrations/tron2/motor_sim.py` | MROS transport and motor torque calculation |
| `capx/integrations/tron2/control_cgn.py`, `grasp_feasibility.py`, `grasp_quality.py` | Strict CGN skill, path checks, and pose ranking |
| `capx/assets/tron2/DACH_TRON2A/xml/scene_table_cube.xml` | CaP-X tabletop scene |
| `env_configs/tron2/` | Strict CGN, CGN multi-turn, and privileged diagnostics profiles |
| `scripts/run_tron2_route2.sh` | Full, isolated Route 2 Oracle/CGN launcher for Linux |
| `scripts/sweep_tron2_cgn.py` | Selection and recording of each candidate trial |
| `scripts/tron2_motor_sim.py` | Standalone model and motor-bus diagnostics |

All TRON2 profiles use the `tron2_cube_lift_low_level` environment. The tabletop
scene uses elliptic contact cones and `impratio=10`. CGN candidates are ranked from
the perception data. The selected motion is planned from the measured arm state
when requested. MuJoCo resolves table contact, and the action log reports each
loaded link with its peak normal load and contact duration. Run the standalone
motor probe in its own test session because it also publishes motor state.

`goto_pose_left()` accepts a position and WXYZ quaternion in the `base_Link`
frame. It uses the measured orientation if the quaternion is omitted. Pass a
positive `approach_offset` for a staged approach or omit it for one direct move.
By default, local IK plans the path and ServoJ streams the time-scaled joint
trajectory at 300 Hz. Set `CAPX_TRON2_LEFT_MOTION=movej` to execute the local IK
waypoints with MoveJ or `CAPX_TRON2_LEFT_MOTION=movep` to use official Cartesian
motion. All three backends verify the measured TCP after requests that reach
the final target check. The ServoJ log records the actual send rate and original
controller notifications.

The local planner diagnoses endpoints that fail its bounded IK search and
refines large joint steps along the path. During ServoJ execution, the tabletop
simulator monitors table contact and can interrupt the stream and send one hold
target. This protection uses simulated contact forces.

## Worktrees and runtime files

| Host and location | Use |
| --- | --- |
| Mac `~/Documents/ChatGPT/Cap-X/cap-x` | Upstream `main`, including local reproduction changes |
| Mac `~/Documents/ChatGPT/Cap-X/tron2-sim` | TRON2 development worktree, backed by the same Git repo as `cap-x` |
| 4090w `~/btian/cap-x` | Upstream reproduction checkout, currently owns the shared Python environment and SAM3 cache |
| 4090w `~/btian/tron2-sim` | Matching TRON2 worktree, backed by the same Git repo as `cap-x` |
| 4090w `~/btian/tron2_manipulation_controller` | LimX controller run pack |
| 4090w `~/btian/tron2-route2-runtime` | SDK and isolated controller fixture |
| 4090w `~/btian/capx-qwen` | Qwen serving environment and checkpoint |
| 4090w `~/btian/capx-tron2-runs/` | New TRON2 run outputs, with one directory per run |

Both `tron2-sim` worktrees use `codex/tron2-single-world`. Keep the Mac and
Linux checkouts on the same revision. The old `cap-x-tron2` worktree and
temporary trial folders on 4090w contain historical changes and recordings. The outer Mac `Cap-X`
folder is a workspace container. Run Git inside its child worktrees.

The 4090w trial currently uses the Python environment from the upstream checkout.
The launcher places the selected TRON2 checkout first on `PYTHONPATH` so the
upstream editable install cannot select the wrong code. For regular use, give the
TRON2 checkout a dedicated Linux environment and point `TRON2_PYTHON` to it.

## Assets and dependencies

- The Git checkout provides the tabletop scene XML. Stage the LimX
  `xml/robot_grasper.xml` and `meshes/` from
  `robot_description/tron2a/DACH_TRON2A/` in the controller pack under
  `capx/assets/tron2/DACH_TRON2A/`. Symlinks are suitable for a shared asset source.
- The Linux Python environment needs MuJoCo, SAM3, Contact-GraspNet, and the tested
  `limxsdk 4.1.2` wheel. Set `TRON2_SDK_DIR` if the wheel is unpacked outside the
  environment. The CGN server resolves the model at
  `capx/third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt`.
  Populate that path and the SAM3 cache before launching inside the private network.
- The test `/opt` fixture contains the simulated controller configuration. Keep it
  beside the SDK and controller pack. The launcher mounts it only inside the
  private mount namespace. On 4090w it lives at `~/btian/tron2-route2-runtime/opt`.

With the model assets staged, a local MuJoCo installation can check the model
without connecting to the controller or loading CUDA models.

```bash
.venv/bin/python scripts/tron2_motor_sim.py --check-model
```

## Run a recorded CGN evaluation on 4090w

Allocate a GPU through `CUDA_VISIBLE_DEVICES` and run from the `tron2-sim`
worktree.

```bash
runtime="$HOME/btian/tron2-route2-runtime"
vendor="$HOME/btian/tron2_manipulation_controller"
run="$HOME/btian/capx-tron2-runs/$(date +%Y%m%d)/route2-$(date +%H%M%S)"
cd "$HOME/btian/tron2-sim"
TRON2_SDK_DIR="$runtime/sdk" bash scripts/run_tron2_route2.sh \
  "$vendor" "$runtime/opt" "$run" --fresh-trials 1
```

Use `--ranks 0 1 2` for three candidates in one batch or
`--ranks 0 --candidate-file /path/to/candidates.npz` for replay. The launcher
isolates networking, mounts and IPC, starts tabletop motor feedback, then starts
the official controller. It stores controller and simulator logs at the run root
and candidate data, video, and results under `evaluation/`. Copy selected outputs
to the Mac for inspection.

## Run the S1–S4 and M1–M4 model profiles

Start an OpenAI-compatible model server on the host loopback and pass its chat
endpoint to the isolated launcher. The default coding model ID is
`checkpoints/Qwen2.5-Coder-7B-Instruct`. S1 uses privileged state. S2
uses RGB-D perception with high-level actions. S3 and S4 use low-level primitives
with and without usage examples, respectively. M1 adds console feedback, M2 adds
RGB images for the coding model, and M3 and M4 use textual visual differencing
with high-level and low-level actions, respectively.

```bash
TRON2_SDK_DIR="$runtime/sdk" bash scripts/run_tron2_route2.sh \
  "$vendor" "$runtime/opt" "$run" --tier M3 \
  --server-url http://127.0.0.1:18089/v1/chat/completions \
  --vdm-model openai/codex-cli-vdm --vdm-model-is-vlm \
  --vdm-server-url "http://127.0.0.1:$vdm_port/v1/chat/completions" \
  --first-seed 1 --cube-xy-jitter-m 0.02 --trials 20 --max-code-blocks 4
```

For this example, expose the authenticated Mac Codex VDM bridge on the host
loopback through SSH reverse forwarding. Set `vdm_port` to the forwarded port
and pass the bridge token in `TRON2_VDM_API_KEY`. M2 needs a separate
vision-capable coding model selected with `--model` and `--model-is-vlm`.

Use a fresh output directory per tier, and keep the seed range and cube reset
jitter identical for paired runs. The VDM can use the coding endpoint or be set
separately with `--vdm-model` and `--vdm-server-url`. Replay of a saved CGN
batch through `--candidate-file` is available for high-level vision diagnostics.
`evaluation/results.json` records the effective profile, seed, reward, execution
status and video path for each episode. The per-episode artifacts include
per-turn execution metrics, generated code, feedback and video.

## Run paired RGB-D evaluations from the Mac

The [Mac launcher guide](../codex-vdm-mac/TRON2.md) covers the RGB-D-only
M1–M3 profiles, Qwen serving, the live dashboard, and how to resume a batch.
The model sees measured TCP feedback and a neutral interruption status.
Physical contact details are available in the raw simulator logs.
