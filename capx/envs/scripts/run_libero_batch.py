import os
# Force weights_only=False for PyTorch loading of legacy files
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
import yaml
import pathlib

import tyro
from capx.envs.launch import LaunchArgs
from capx.envs.launch import main as launch_main

# Import Libero to discover tasks
try:
    from libero import benchmark
except ImportError:
    print("Error: Libero not found. Make sure it is installed and PYTHONPATH is set.")
    sys.exit(1)


@dataclass
class LiberoBatchLaunchArgs:
    """Command-line arguments for automated Libero batch execution."""

    # Base configuration file
    base_config_path: str = "env_configs/libero/franka_libero.yaml"

    # Suites to run
    suites: list[str] = field(
        default_factory=lambda: [
            # Table 2 suites: Pos (swap) and Task perturbations
            "libero_object_swap",
            "libero_object_task",
            "libero_goal_swap",
            "libero_goal_task",
            "libero_spatial_swap",
            "libero_spatial_task",
        ]
    )

    # Models to run (copied from run_batch.py default)
    models: list[str] = field(
        default_factory=lambda: [
            # "openai/gpt-5.2",
            "google/gemini-3.1-pro-preview"
        ]
    )

    server_url: str = "http://127.0.0.1:8110/chat/completions"  # local NV inference server

    # Output directory base
    output_dir: str = "./outputs/libero_batch_run"

    # Other LaunchArgs overrides
    temperature: float = 1.0
    max_tokens: int = 2048 * 10
    reasoning_effort: str = "medium"
    api_key: str | None = None
    use_visual_feedback: bool | None = None
    use_img_differencing: bool | None = None
    use_legacy_multi_turn_decision_prompt: bool | None = None
    total_trials: int | None = None
    num_workers: int | None = None
    record_video: bool | None = None
    debug: bool = False
    use_oracle_code: bool | None = None


def main(args: LiberoBatchLaunchArgs) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    
    # Load base configuration
    if not os.path.exists(args.base_config_path):
        print(f"Error: Base config file not found: {args.base_config_path}")
        sys.exit(1)
        
    with open(args.base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    tasks_to_run = []
    
    print(f"Collecting tasks for suites: {args.suites}")
    
    for suite_name in args.suites:
        if suite_name not in benchmark_dict:
            print(f"Warning: Suite '{suite_name}' not found in Libero benchmarks.")
            continue
            
        task_suite = benchmark_dict[suite_name]()
        num_tasks = task_suite.n_tasks
        
        print(f"Suite {suite_name} has {num_tasks} tasks.")
        
        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            task_name = task.name
            
            # Determine number of initial states (trials)
            try:
                # We force weights_only=False because Libero checkpoints might be old
                import torch
                # Temporarily bypass potential weights_only restrictions if feasible, or catch warning
                # Note: os.environ is global, set it at script start or ensure it's handled.
                # Here we trust the environment or the user to have set it if needed, 
                # but we can try to load to check count.
                # Actually, get_task_init_states uses torch.load internally.
                # We need to rely on the environment variable being set if PyTorch requires it.
                init_states = task_suite.get_task_init_states(task_id)
                assert init_states is not None, f"No initial states found for task {task_name}"
                num_trials = len(init_states)
            except Exception as e:
                print(f"Warning: Could not determine init states for {task_name}, defaulting to 50. Error: {e}")
                num_trials = 50

            tasks_to_run.append((suite_name, task_id, task_name, num_trials))

    print(f"Total tasks to run: {len(tasks_to_run)}")
    
    total_runs = len(args.models) * len(tasks_to_run)
    print(f"Total experimental runs (models x tasks): {total_runs}")
    
    experiment_idx = 1
    failed_runs = []
    
    for model in args.models:
        print(f"\n{'=' * 80}")
        print(f"Running model: {model}")
        print(f"{'=' * 80}")
        
        for suite_name, task_id, task_name, num_trials in tasks_to_run:
            print(f"\n{'-' * 80}")
            print(f"Running Experiment {experiment_idx}/{total_runs}")
            print(f"Model: {model}")
            print(f"Suite: {suite_name}, Task ID: {task_id}, Task Name: {task_name}")
            print(f"Trials: {num_trials}")
            print(f"{'-' * 80}\n")
            
            # Construct config from base template
            config = base_config.copy()
            
            # Deep copy to ensure we don't modify the shared base_config for subsequent runs
            # (although yaml.load usually returns new dict, copy is shallow, deepcopy is safer)
            import copy
            config = copy.deepcopy(base_config)

            # Inject task details
            # We overwrite the 'low_level' entry to be a dictionary definition for FrankaLiberoEnv
            # instead of the string reference (e.g. 'franka_libero_pick_place_low_level')
            
            # Check if env/cfg/privileged exists, use it if so
            is_privileged = config.get("env", {}).get("cfg", {}).get("privileged", False)
            
            config["env"]["cfg"]["low_level"] = {
                "_target_": "capx.envs.simulators.libero.FrankaLiberoEnv",
                "suite_name": suite_name,
                "task_id": task_id,
                "privileged": is_privileged,
                "max_steps": 8000,
                "seed": None,
                "enable_render": True,
                "viser_debug": False
            }
            
            # Set trials count dynamically
            config["trials"] = num_trials
            
            # Customize output directory
            # Structure: output_dir/suite_name/task_name/run
            # launch.py will append model_name automatically to the last component if we rely on its default behavior
            # But we want: output_dir/suite/task/model/run (or similar)
            
            # To get output_dir/suite/task/model/run:
            # We set config["output_dir"] = .../suite/task/run
            # launch.py splits: [..., suite, task, run]
            # Inserts model at -1: [..., suite, task, model, run]
            # Result: .../suite/task/model/run
            
            config_dir_base = os.path.join(os.path.abspath(args.output_dir), suite_name, task_name)
            config["output_dir"] = os.path.join(config_dir_base, "run")
            
            # Create the base directory if it doesn't exist to save the config
            os.makedirs(config_dir_base, exist_ok=True)
            
            # Save config permanently to the run directory (parent of "run" subfolder effectively)
            config_filename = "config.yaml"
            config_path = os.path.join(config_dir_base, config_filename)
            
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            
            # Create LaunchArgs
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
                # We do NOT pass output_dir here to avoid the logic that uses config_stem
                output_dir=None, 
                debug=args.debug,
                use_oracle_code=args.use_oracle_code,
            )
            
            try:
                launch_main(launch_args)
            except Exception as e:
                print(f"\nERROR running {suite_name}/{task_name} with model {model}: {e}")
                traceback.print_exc()
                failed_runs.append((model, suite_name, task_name))
            
            experiment_idx += 1

    if failed_runs:
        print(f"\n{'=' * 80}")
        print(f"Batch execution completed with {len(failed_runs)} failures:")
        for model, suite, task in failed_runs:
            print(f"  - model={model}, suite={suite}, task={task}")
        print(f"{'=' * 80}")
        sys.exit(1)
    else:
        print(f"\n{'=' * 80}")
        print("Batch execution completed successfully.")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    tyro.cli(main)
