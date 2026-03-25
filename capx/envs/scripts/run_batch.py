import os
import sys
from dataclasses import dataclass, field

import tyro
from capx.envs.launch import LaunchArgs
from capx.envs.launch import main as launch_main

# Set default environment variable for MuJoCo, similar to launch.py
os.environ.setdefault("MUJOCO_GL", "egl")


@dataclass
class BatchLaunchArgs:
    """Command-line arguments for batch execution of CaP-X environments."""

    # List of YAML config paths to run
    config_paths: list[str] = field(
        default_factory=lambda: [
            # "env_configs/two_arm_handover/hillclimb/debug_multimodel_two_arm_handover_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_handover/hillclimb/ensemble_two_arm_handover_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_handover/hillclimb/multimodel_two_arm_handover_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_handover/hillclimb/two_arm_handover_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_handover/hillclimb/two_arm_handover_multiturn_vdm_reduced_api.yaml"

            # "env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml",
            "env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml",
            # "env_configs/cube_lifting/franka_robosuite_cube_lifting_reduced_api.yaml",
            # "env_configs/cube_lifting/franka_robosuite_cube_lifting_reduced_api_exampleless.yaml",
            # "env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml",
            "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
            # "env_configs/cube_stack/franka_robosuite_cube_stack_reduced_api.yaml",
            # "env_configs/cube_stack/franka_robosuite_cube_stack_reduced_api_exampleless.yaml",
            # "env_configs/cube_restack/franka_robosuite_cube_restack_privileged.yaml",
            # "env_configs/cube_restack/franka_robosuite_cube_restack.yaml",
            # "env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml",
            "env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml",
            # "env_configs/spill_wipe/franka_robosuite_spill_wipe_reduced_api.yaml",
            # "env_configs/spill_wipe/franka_robosuite_spill_wipe_reduced_api_exampleless.yaml",
            # "env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml",
            "env_configs/nut_assembly/franka_robosuite_nut_assembly.yaml",
            # "env_configs/nut_assembly/franka_robosuite_nut_assembly_reduced_api.yaml",
            # "env_configs/nut_assembly/franka_robosuite_nut_assembly_reduced_api_exampleless.yaml",
            # "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_privileged.yaml",
            "env_configs/two_arm_lift/franka_robosuite_two_arm_lift.yaml",

            # "env_configs/two_arm_handover/two_arm_handover_multiturn_vdm.yaml",
            # "env_configs/two_arm_handover/two_arm_handover_multiturn_vdm_legacy.yaml",
            # "env_configs/two_arm_handover/two_arm_handover_multiturn.yaml",
            # "env_configs/two_arm_handover/two_arm_handover_privileged.yaml",
            "env_configs/two_arm_handover/two_arm_handover.yaml",
            # "env_configs/two_arm_handover/two_arm_handover_reduced.yaml",
            # "env_configs/two_arm_handover/two_arm_handover_reduced_exampleless.yaml",
            # "env_configs/two_arm_handover/two_arm_handover_multiturn_vf.yaml",

            # "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn.yaml",
            # "env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn.yaml",  # We run multiturn last since they take the longest to run
            # "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn.yaml",
            # "env_configs/cube_restack/franka_robosuite_cube_restack_multiturn.yaml",
            # "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn.yaml",
            # "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn.yaml",
            # "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml",
            # "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vf.yaml",

            # "env_configs/cube_restack/franka_robosuite_cube_restack_multiturn_vdm.yaml",
            # "env_configs/cube_restack/franka_robosuite_cube_restack_multiturn_vf.yaml",

            # "env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm.yaml",
            # "env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vf.yaml",
            # "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm.yaml",
            # "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vf.yaml",
            # "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm.yaml",
            # "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vf.yaml",
            # "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vdm.yaml",
            # "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vf.yaml",
            # Hill climbing VDM-L
            # "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_reduced_api.yaml",
            # "env_configs/cube_restack/franka_robosuite_cube_restack_multiturn_vdm_reduced_api.yaml",
            # "env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm_reduced_api.yaml",
            # "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm_reduced_api.yaml",
            # "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm_reduced_api.yaml",
            # "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api.yaml",

            # Hill climbing VDM-L-SL
            # "env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_restack/franka_robosuite_cube_restack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api_skill_lib.yaml"
            
            # multimodel: Hill climb 3 model debug
            # "env_configs/cube_lifting/hillclimb/multimodel_franka_robosuite_cube_lifting_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_stack/hillclimb/multimodel_franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_restack/hillclimb/multimodel_franka_robosuite_cube_restack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/nut_assembly/hillclimb/multimodel_franka_robosuite_nut_assembly_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/spill_wipe/hillclimb/multimodel_franka_robosuite_spill_wipe_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_lift/hillclimb/multimodel_franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_handover/hillclimb/multimodel_franka_robosuite_two_arm_handover_multiturn_vdm_reduced_api_skill_lib.yaml",

            # debug multimodel: Hill climb 3 model debug parallel
            # "env_configs/cube_lifting/hillclimb/debug_multimodel_franka_robosuite_cube_lifting_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_stack/hillclimb/debug_multimodel_franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_restack/hillclimb/debug_multimodel_franka_robosuite_cube_restack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/nut_assembly/hillclimb/debug_multimodel_franka_robosuite_nut_assembly_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/spill_wipe/hillclimb/debug_multimodel_franka_robosuite_spill_wipe_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_lift/hillclimb/debug_multimodel_franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_handover/hillclimb/debug_multimodel_franka_robosuite_two_arm_handover_multiturn_vdm_reduced_api_skill_lib.yaml",

            # ensemble: Hill climb 1 model parallel
            # "env_configs/cube_lifting/hillclimb/ensemble_franka_robosuite_cube_lifting_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_stack/hillclimb/ensemble_franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_restack/hillclimb/ensemble_franka_robosuite_cube_restack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/nut_assembly/hillclimb/ensemble_franka_robosuite_nut_assembly_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/spill_wipe/hillclimb/ensemble_franka_robosuite_spill_wipe_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_lift/hillclimb/ensemble_franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_handover/hillclimb/ensemble_franka_robosuite_two_arm_handover_multiturn_vdm_reduced_api_skill_lib.yaml",

            # debug ensemble: Hill climb 1 model parallel debug
            # "env_configs/cube_lifting/hillclimb/debug_ensemble_franka_robosuite_cube_lifting_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_stack/hillclimb/debug_ensemble_franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/cube_restack/hillclimb/debug_ensemble_franka_robosuite_cube_restack_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/nut_assembly/hillclimb/debug_ensemble_franka_robosuite_nut_assembly_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/spill_wipe/hillclimb/debug_ensemble_franka_robosuite_spill_wipe_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_lift/hillclimb/debug_ensemble_franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api_skill_lib.yaml",
            # "env_configs/two_arm_handover/hillclimb/debug_ensemble_franka_robosuite_two_arm_handover_multiturn_vdm_reduced_api_skill_lib.yaml",

            # LIBERO
            # "env_configs/libero_pick_place/franka_libero_pick_place.yaml",
            "env_configs/libero_pick_place/franka_libero_pick_place_vdm_reduced_skill_library.yaml",
        ]
    )
    """List of paths to the YAML configuration files to run sequentially."""

    # Overrides (mirrored from LaunchArgs to allow global overrides)

    # server_url: str = "http://128.32.175.97:8111/v1/chat/completions" # athena2 vllm
    # server_url: str = "http://128.32.175.74:8111/v1/chat/completions" # duchamp3 vllm
    # server_url: str = "http://128.32.175.42:8111/v1/chat/completions"  # macmini vllm server
    # server_url: str = "http://128.32.175.42:8110/chat/completions"  # macmini nv server jump
    # server_url: str = "http://0.0.0.0:8009/v1/responses"  # local nv server for running codex models
    # server_url: str = "http://0.0.0.0:8110/chat/completions"  # local nv server for running open source models
    server_url: str = "http://127.0.0.1:8110/chat/completions"  # local nv server jump


    models: list[str] = field(
        default_factory=lambda: [
            # "nvidia/openai/gpt-oss-120b", # Open source models
            # "Qwen/Qwen2.5-Coder-7B-Instruct",
            # "openai/gpt-oss-20b",
            # "nvdev/deepseek-ai/deepseek-v3-0324",
            # "deepseek-ai/deepseek-r1-0528",
            # "deepseek-ai/deepseek-v3.1",
            # "nvdev/deepseek-ai/deepseek-r1",
            # "nvdev/qwen/qwen-235b",
            # "moonshotai/kimi-k2-instruct-0905",
            # "moonshotai/kimi-k2-instruct",
            # "deepseek-ai/deepseek-v3.1-terminus",
            # "Qwen/Qwen2.5-Coder-7B-Instruct",
            # "Qwen/Qwen3-VL-8B-Instruct",
            "google/gemini-3.1-pro-preview", # Closed source models
            # "google/gemini-2.5-flash-lite",
            # "aws/anthropic/claude-haiku-4-5-v1",
            # "aws/anthropic/claude-opus-4-5",
            # "openai/openai/gpt-5.2",
            # "openai/openai/gpt-5.1"
            # "azure/openai/o1",
            # "azure/openai/o4-mini",
            # "azure/openai/gpt-5.1-codex",
            # "azure/openai/gpt-5.1-codex",
        ]
    )
    """Names of the models to query on the vLLM server."""

    temperature: float = 1.0
    """Sampling temperature for code generation (higher = more random)."""

    max_tokens: int = 2048 * 10
    """Maximum number of tokens to generate in the model response."""

    reasoning_effort: str = "medium"
    """Effort level for reasoning models (if applicable)."""

    api_key: str | None = None
    """Optional API key for authentication with the model server."""

    use_visual_feedback: bool | None = None
    """Whether to provide visual feedback (images) to the model during generation."""

    use_img_differencing: bool | None = None
    """Whether to use image differencing."""

    use_legacy_multi_turn_decision_prompt: bool | None = None
    """Whether to use the legacy multi-turn decision prompt."""

    total_trials: int | None = None
    """Total number of trials to run per yaml config file. Overrides the value in the YAML config."""

    num_workers: int | None = None
    """Number of parallel worker processes to use. Overrides the value in the YAML config."""

    record_video: bool | None = None
    """Whether to record and save videos of the environment execution."""

    output_dir: str | None = None
    """Directory to save trial outputs (code, logs, videos)."""

    debug: bool = False
    """Enable debug logging (prints full model responses)."""

    use_oracle_code: bool | None = None
    """If True, uses pre-defined oracle code instead of querying the model."""


def main(args: BatchLaunchArgs) -> None:
    """Run multiple experiments sequentially."""

    total_runs = len(args.models) * len(args.config_paths)
    print(
        f"Found {len(args.models)} models x {len(args.config_paths)} configurations "
        f"to run ({total_runs} total)."
    )

    failed_runs = []
    experiment_idx = 1

    for model in args.models:
        print(f"\n{'=' * 80}")
        print(f"Running model: {model}")
        print(f"{'=' * 80}")

        for config_path in args.config_paths:
            print(f"\n{'-' * 80}")
            print(f"Running Experiment {experiment_idx}/{total_runs}")
            print(f"Model: {model}")
            print(f"Config: {config_path}")
            print(f"{'-' * 80}\n")

            try:
                # If output_dir is overridden, append model and config name to avoid
                # collisions and ensure unique paths for each config/model in the batch.
                current_output_dir = args.output_dir
                if current_output_dir:
                    import pathlib

                    config_stem = pathlib.Path(config_path).stem
                    model_dir = model.replace("/", "_")
                    current_output_dir = os.path.join(current_output_dir, model_dir, config_stem)

                # Create LaunchArgs with the current config_path and global overrides
                launch_args = LaunchArgs(
                    config_path=config_path,
                    server_url=args.server_url,
                    model=model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    api_key=args.api_key,
                    use_visual_feedback=args.use_visual_feedback,
                    use_img_differencing=args.use_img_differencing,
                    use_legacy_multi_turn_decision_prompt=args.use_legacy_multi_turn_decision_prompt,
                    total_trials=args.total_trials,
                    num_workers=args.num_workers,
                    record_video=args.record_video,
                    output_dir=current_output_dir,
                    debug=args.debug,
                    use_oracle_code=args.use_oracle_code,
                )

                launch_main(launch_args)

            except Exception as e:
                print(f"\nERROR running config {config_path} with model {model}: {e}")
                import traceback

                traceback.print_exc()
                failed_runs.append((model, config_path))

            experiment_idx += 1

    if failed_runs:
        print(f"\n{'=' * 80}")
        print(f"Batch execution completed with {len(failed_runs)} failures:")
        for model, cfg in failed_runs:
            print(f"  - model={model}, config={cfg}")
        print(f"{'=' * 80}")
        sys.exit(1)
    else:
        print(f"\n{'=' * 80}")
        print("Batch execution completed successfully.")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    tyro.cli(main)
