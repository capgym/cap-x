# CaP-X

CaP-X is a framework for using LLM/VLM-generated code as robot control policies. It provides a Gymnasium environment that accepts Python code strings as actions, executes them in a sandbox with curated tool APIs, and returns observations and rewards. The system supports both simulated (robosuite, MuJoCo, LIBERO) and real Franka Panda robots.

## Key Commands

```bash
# Run an evaluation benchmark
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model <model-name> --total-trials 10

# Run with the interactive web UI
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --web-ui

# Run tests
uv run pytest tests/

# Lint
uv run ruff check capx/
```

## Architecture

```
capx/
  envs/           # Gymnasium environments
    base.py       # BaseEnv (low-level sim/real env interface)
    tasks/        # CodeExecutionEnvBase and task-specific wrappers
      base.py     # CodeExecEnvConfig, SimpleExecutor, CodeExecutionEnvBase
      franka/     # Franka Panda task environments
    configs/      # YAML config loading and Hydra-style instantiate()
    simulators/   # Low-level simulator wrappers (robosuite, MuJoCo, LIBERO)
    launch.py     # CLI entry point (LaunchArgs, trial orchestration)
    runner.py     # Parallel/sequential trial runners
    trial.py      # Single-trial execution loop with multi-turn regeneration
  integrations/   # Robot control APIs (ApiBase subclasses)
    base_api.py   # ApiBase ABC, API registry (register_api/get_api)
    franka/       # Franka-specific APIs (control, grasping, etc.)
    r1pro/        # R1 Pro robot APIs
  llm/            # LLM/VLM client (query_model, ensemble queries)
  skills/         # Evolving skill library (extract, store, inject)
  serving/        # API servers (SAM3, ContactGraspNet, PyRoKI)
  utils/          # Launch utilities, execution logger
  web/            # Interactive web UI (FastAPI + frontend)
env_configs/      # YAML configs per task (cube_stack, nut_assembly, etc.)
tests/            # Integration and unit tests
```

## How It Works

1. A YAML config (in `env_configs/`) specifies a `CodeExecutionEnvBase` with a low-level env and a list of API names.
2. `instantiate()` builds the env, which creates API instances via the `register_api`/`get_api` registry.
3. The agent (LLM) receives a prompt with API docs and generates Python code.
4. `SimpleExecutor` runs the code with access to `env` (low-level) and `APIS` (dict of ApiBase instances).
5. Multi-turn: the system captures visual feedback, asks the model to REGENERATE or FINISH, and loops.

## Claude Code Integration

Claude Code can be used as an autonomous research agent to hillclimb task success rates. The workflow:

1. **Run evaluations** via `/run-eval` to establish baselines
2. **Analyze failures** by reading trial outputs and code
3. **Improve prompts** in YAML configs and task `.py` files
4. **Evolve the skill library** via `/skill-library` — extract reusable functions from successful trials
5. **Re-evaluate** to measure improvement

### Slash Commands

- `/run-eval` -- Run a CaP-X evaluation benchmark
- `/skill-library` -- Show and manage the evolving skill library
- `/regression-test` -- Run CaP-X regression tests

### Skill Library

The evolving skill library persists at `.capx_skills.json` in the project root. It stores reusable Python functions extracted from successful trials.

- **Source code**: `capx/skills/library.py` (SkillLibrary), `capx/skills/extractor.py` (extract_functions)
- **Slash command**: `/skill-library` to manage skills from Claude Code

### Autoresearch Loop

For autonomous experimentation, Claude Code can:

- Modify prompts in YAML configs and task `.py` files
- Add new skills to `.capx_skills.json`
- Change model, temperature, multi-turn settings via CLI flags
- Modify API implementations in `capx/integrations/franka/`
- Run parallel evaluations across multiple GPUs
- Track results and iterate

See [docs/claude-code-integration.md](docs/claude-code-integration.md) for the full autoresearch setup.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run a specific test
uv run pytest tests/test_environments.py -v

# Run with timeout
uv run pytest tests/ --timeout=120
```

## Coding Conventions

- Python 3.10+ with type annotations (use `X | Y` union syntax, not `Union[X, Y]`).
- Google-style docstrings for all public functions.
- Ruff for linting: `line-length = 100`, `target-version = "py312"`.
- Third-party code lives in `capx/third_party/` and is excluded from linting.
- New APIs subclass `ApiBase` and register via `register_api()` in their module.
- Environment configs use `_target_` keys for Hydra-style lazy instantiation.
- Use `uv run` for all commands to ensure the correct virtual environment.
