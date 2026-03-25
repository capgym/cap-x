import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from openai import OpenAI

API_KEY = os.environ.get("API_KEY")

SYS_PROMPT = """
You are an expert model behavior analyst. 
Your task is to meticulously analyze the trace of a large language model code generation failures and group the failed trials into categories based on the root causes of the failures.
For each category, you should provide a concise description of the root cause of the failures, and a list of the failed trials.
The successful trials are also included for reference.
Focus on high level mistakes. Classify low level mistakes into high level, broad categories.
Whenever you mention a type of model behavior, you must cite the trial and excerpt of the code demonstrating this behavior.
YOU MUST GUARANTEE THAT YOUR OUTPUTS ARE CORRECT AND DO NOT HALLUCINATE.
"""
SYS_PROMPT2 = """
You are an expert model behavior analyst. 
Your task is to meticulously analyze the trace of a large language model code generation and elevate and highlight the most important and interesting code generation behaviors that are worth further discussion.
Make case studies of these behaviors, including example code generation traces. 
Whenever you mention a type of model behavior, you must cite the trial and excerpt of the code demonstrating this behavior.
YOU MUST GUARANTEE THAT YOUR OUTPUTS ARE CORRECT AND DO NOT HALLUCINATE.
"""


@dataclass
class TrialData:
    """Data class representing the results of a single trial."""

    trial_folder_path: Path
    sandbox_rc: int
    reward: float
    task_completed: bool
    initial_prompt_txt_path: Path
    summary_txt: Path


class ExperimentParser:
    """
    Parses experiment results from a structured directory.

    The expected directory structure is:
    experiment_root/
        initial_prompt.txt
        trial_{idx}_sandboxrc_{rc}_reward_{rew}_taskcompleted_{comp}/
            all_responses.json
            ...
    """

    # Regex to parse the trial directory name
    # Example: trial_100_sandboxrc_0_reward_0.952_taskcompleted_1
    TRIAL_DIR_PATTERN = re.compile(
        r"trial_(\d+)_sandboxrc_(\d+)_reward_([-\d\.]+)_taskcompleted_(\d+)"
    )

    def __init__(self, experiment_dir: str | Path):
        self.experiment_dir = Path(experiment_dir)
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")

    def parse_trials(self) -> dict[int, TrialData]:
        """
        Parses the experiment directory and returns a mapping from trial index to TrialData.

        Returns:
            Dict[int, TrialData]: A dictionary where keys are trial indices and values are TrialData objects.
        """
        results: dict[int, TrialData] = {}
        initial_prompt_path = self.experiment_dir / "initial_prompt.txt"

        # Iterate over all items in the experiment directory
        for item in self.experiment_dir.iterdir():
            if not item.is_dir():
                continue

            match = self.TRIAL_DIR_PATTERN.match(item.name)
            if match:
                trial_idx = int(match.group(1))
                sandbox_rc = int(match.group(2))
                reward = float(match.group(3))
                task_completed = bool(int(match.group(4)))

                trial_data = TrialData(
                    trial_folder_path=item.absolute(),
                    sandbox_rc=sandbox_rc,
                    reward=reward,
                    task_completed=task_completed,
                    initial_prompt_txt_path=initial_prompt_path.absolute(),
                    summary_txt=(item / "summary.txt").absolute(),
                )
                results[trial_idx] = trial_data

        # Sort by trial index for consistent ordering in the dict (Python 3.7+ dicts are insertion ordered)
        sorted_results = dict(sorted(results.items()))
        return sorted_results


def compose_failures(sorted_results: dict[int, TrialData]) -> str:
    """
    concatenate all the failures code generation attempts together into a single string
    """
    failures = []
    for trial_idx, trial_data in sorted_results.items():
        if not trial_data.task_completed:
            failed_code = trial_data.summary_txt.read_text()
            failures.append(f"Trial {trial_idx}:\n{failed_code}")
    return "\n\n".join(failures)


def compose_successes(sorted_results: dict[int, TrialData]) -> str:
    """
    concatenate all the successes code generation attempts together into a single string
    """
    successes = []
    for trial_idx, trial_data in sorted_results.items():
        if trial_data.task_completed:
            successful_code = trial_data.summary_txt.read_text()
            successes.append(f"Trial {trial_idx}:\n{successful_code}")
    return "\n\n".join(successes)


def analyze_failures(sorted_results: dict[int, TrialData]) -> str:
    task_description = next(iter(sorted_results.values())).initial_prompt_txt_path.read_text()
    # print("task description: ", task_description)
    failures = compose_failures(sorted_results)
    successes = compose_successes(sorted_results)
    prompt = f"""
    {SYS_PROMPT}
    Task description:
    {task_description}
    Successes:
    {successes}
    Failures:
    {failures}
    """
    client = OpenAI(base_url="https://inference-api.nvidia.com/v1/", api_key=API_KEY)
    completion = client.chat.completions.create(
        model="google/gemini-3.1-pro-preview",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def analyze_highlights(sorted_results: dict[int, TrialData]) -> str:
    task_description = next(iter(sorted_results.values())).initial_prompt_txt_path.read_text()
    # print("task description: ", task_description)
    failures = compose_failures(sorted_results)
    successes = compose_successes(sorted_results)
    prompt = f"""
    {SYS_PROMPT2}
    Task description:
    {task_description}
    Successes:
    {successes}
    Failures:
    {failures}
    """
    client = OpenAI(base_url="https://inference-api.nvidia.com/v1/", api_key=API_KEY)
    completion = client.chat.completions.create(
        model="google/gemini-3.1-pro-preview",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content
