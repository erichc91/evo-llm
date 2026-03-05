# reasoning.py — ARC-Easy reasoning task loader and scorer

import json
import re
from pathlib import Path

TASK_NAME = "reasoning"
SEED_PROMPT = "You are a careful scientific reasoner. Read each question and select the best answer."

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "arc_easy_sample.json"


def load_task(data_path=None) -> list[dict]:
    """
    IN:  optional path to JSON file (str or Path); defaults to data/arc_easy_sample.json
    PROC: load file, validate each entry has id/question/choices/answer fields
    OUT: list of question dicts
    """
    path = Path(data_path) if data_path else _DEFAULT_DATA_PATH

    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    required = {"id", "question", "choices", "answer"}
    for q in questions:
        missing = required - q.keys()
        if missing:
            raise ValueError(f"Question {q.get('id', '?')} missing fields: {missing}")

    return questions


def format_prompt(question_dict: dict) -> str:
    """
    IN:  one question dict with 'question' and 'choices' keys
    PROC: format as a multiple-choice prompt string
    OUT: formatted prompt str
    """
    q = question_dict["question"]
    choices = question_dict["choices"]
    lines = [f"Question: {q}"]
    for letter in ("A", "B", "C", "D"):
        if letter in choices:
            lines.append(f"{letter}) {choices[letter]}")
    lines.append("Answer with ONLY the letter (A, B, C, or D).")
    return "\n".join(lines)


def score_response(response: str, expected_answer: str) -> float:
    """
    IN:  model response string, expected answer letter ("A"/"B"/"C"/"D")
    PROC: extract first capital letter A-D from response; compare to expected
    OUT: 1.0 if correct, 0.0 if wrong or unparseable
    """
    match = re.search(r"\b([A-D])\b", response.strip())
    if not match:
        # fallback: first bare capital letter A-D anywhere in response
        match = re.search(r"[A-D]", response)
    if not match:
        return 0.0
    return 1.0 if match.group()[-1] == expected_answer.upper() else 0.0
