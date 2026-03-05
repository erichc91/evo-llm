# coding.py — Python coding challenge task loader and scorer

import json
import re
import textwrap
from pathlib import Path

TASK_NAME = "coding"
SEED_PROMPT = "You are an expert Python programmer. Write clean, correct functions. Return ONLY the function definition."

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "coding_challenges.json"


def load_task(data_path=None) -> list[dict]:
    """
    IN:  optional path to JSON file (str or Path); defaults to data/coding_challenges.json
    PROC: load file, validate each entry has id/description/function_name/test_cases fields
    OUT: list of challenge dicts
    """
    path = Path(data_path) if data_path else _DEFAULT_DATA_PATH

    with open(path, "r", encoding="utf-8") as f:
        challenges = json.load(f)

    required = {"id", "description", "function_name", "test_cases"}
    for c in challenges:
        missing = required - c.keys()
        if missing:
            raise ValueError(f"Challenge {c.get('id', '?')} missing fields: {missing}")

    return challenges


def format_prompt(challenge_dict: dict) -> str:
    """
    IN:  one challenge dict with 'description' key
    PROC: wrap description in a standard coding prompt
    OUT: formatted prompt str
    """
    return (
        "Write a Python function with this signature and behavior:\n"
        f"{challenge_dict['description']}\n"
        "Return ONLY the function definition, no explanation, no markdown."
    )


def _extract_code(response: str) -> str:
    """Strip markdown fences and return raw code."""
    # Remove ```python ... ``` or ``` ... ``` blocks
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return response


def score_response(response: str, challenge_dict: dict) -> float:
    """
    IN:  model response string (expected to contain a Python function definition),
         challenge dict with 'function_name' and 'test_cases' keys
    PROC:
        1. Extract code block (strip markdown if present)
        2. exec() the function into a local namespace
        3. Run each test case: call fn(*input), compare to expected
        4. Compute pass_rate = passed / total
        Syntax errors, runtime errors → 0.0 for that test case
    OUT: float 0.0–1.0
    """
    code = _extract_code(response)
    code = textwrap.dedent(code)

    fn_name = challenge_dict["function_name"]
    test_cases = challenge_dict["test_cases"]

    namespace: dict = {}
    try:
        exec(compile(code, "<llm_response>", "exec"), namespace)  # noqa: S102
    except SyntaxError:
        return 0.0

    fn = namespace.get(fn_name)
    if fn is None or not callable(fn):
        return 0.0

    passed = 0
    for tc in test_cases:
        try:
            result = fn(*tc["input"])
            if result == tc["expected"]:
                passed += 1
        except Exception:
            pass  # runtime error counts as failure for this test case

    return passed / len(test_cases) if test_cases else 0.0
