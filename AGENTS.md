# Repository Guidelines

## Project Structure & Module Organization

`capx/` contains the Python package: environments and task definitions live in
`capx/envs/`, reusable robot and vision adapters in `capx/integrations/`, service
entry points in `capx/serving/`, and shared helpers in `capx/utils/`. Keep a robot
integration under its existing namespace, such as `capx/integrations/g1/`.
Task YAML files belong in `env_configs/`; tests mirror source areas in `tests/`
and `tests/integrations/`. Treat `capx/third_party/` as vendored code: avoid
unrelated edits there. Use `tools/` for standalone bringup, calibration, and
debugging utilities.

## Build, Test, and Development Commands

Create the standard environment with `uv sync`; add simulator extras only when
needed, for example `uv sync --extra contactgraspnet`. Run a task locally with:

```bash
uv run --no-sync --active capx/envs/launch.py --config-path <config.yaml>
```

Run a focused test with `uv run pytest tests/test_environments.py -q`; run the
simulator smoke suite with `./scripts/regression_test.sh quick`. Lint and format
repository code using `uv run ruff check capx tests` and `uv run ruff format capx tests`.

## Coding Style & Naming Conventions

Use Python 3.10+ with four-space indentation, type hints on public interfaces,
and short docstrings for non-obvious behavior. Ruff is configured for a 100-column
target; follow its import ordering. Use `snake_case` for functions, variables, and
files, `PascalCase` for classes, and `UPPER_CASE` for constants. New task configs
should use descriptive lowercase YAML names, e.g. `g1_grasp_bottle_left.yaml`.

## Testing Guidelines

Use pytest. Name tests `test_<behavior>` and keep unit tests deterministic; mock
network, model, and robot I/O at the integration boundary. Add a regression test
when changing an API, configuration mapping, or command serialization. For G1
changes, run the focused adapter/config tests before any robot test. Set
`CAPX_G1_DRY_RUN=true` for software-only checks; only Gateway may publish
`rt/lowcmd` during real-robot operation.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects such as `add g1 vision bridge` or
`fix SAM3 device mismatch`. Keep commits scoped and explain behavioral or config
changes in the body. PRs should state the affected environments/configs, tests
run, required assets or services, and safety impact. Include screenshots or output
artifacts for UI, visualization, or perception changes; never commit API keys,
robot credentials, generated outputs, or model weights.
