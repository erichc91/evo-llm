# fitness.py — Evaluate a system prompt's fitness on a task dataset
# Fitness = mean score across n_samples drawn from the task dataset
# Supports: reasoning (exact-match), coding (pytest pass-rate), hybrid (+ llm-judge)

import logging
import random

from src import llm_client
from src.llm_client import OllamaError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge rubrics
# ---------------------------------------------------------------------------

REASONING_RUBRIC = (
    "Score how well the response answers the question. "
    "10 = correct and clear, 5 = partially correct, 0 = wrong or irrelevant."
)
CODING_RUBRIC = (
    "Score the Python function. "
    "10 = correct, clean, handles edge cases. "
    "5 = mostly correct with minor issues. "
    "0 = wrong, syntax error, or doesn't match the signature."
)

_RUBRICS = {
    "reasoning": REASONING_RUBRIC,
    "coding": CODING_RUBRIC,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def get_task_module(task_name: str):
    """
    IN:  "reasoning" or "coding"
    OUT: the task module
    Raises ValueError on unknown task name.
    """
    if task_name == "reasoning":
        from src.tasks import reasoning
        return reasoning
    if task_name == "coding":
        from src.tasks import coding
        return coding
    raise ValueError(f"Unknown task name: {task_name!r}. Expected 'reasoning' or 'coding'.")


def get_seed_prompt(task_name: str) -> str:
    """
    IN:  task_name
    OUT: SEED_PROMPT from the appropriate task module
    """
    return get_task_module(task_name).SEED_PROMPT


def _score_item(task_name: str, task_module, response: str, item: dict) -> float:
    """Route score_response correctly per task (different 2nd-arg contracts)."""
    if task_name == "reasoning":
        return task_module.score_response(response, item["answer"])
    # coding: pass the full challenge dict
    return task_module.score_response(response, item)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_prompt(
    prompt: str,
    task_name: str,
    model: str,
    n_samples: int = 10,
    use_judge: bool = False,
    judge_weight: float = 0.3,
    data_path: str = None,
) -> dict:
    """
    IN:
        prompt       — system prompt being evaluated
        task_name    — "reasoning" or "coding"
        model        — ollama model name
        n_samples    — number of questions to evaluate (randomly sampled)
        use_judge    — add LLM-as-judge score on top of task score
        judge_weight — weight of judge score (0.0 = pure task score)
        data_path    — optional override for default data file path
    PROC:
        1. Load task dataset via tasks/*.load_task()
        2. Randomly sample n_samples questions (or all if dataset < n_samples)
        3. For each question:
           a. Format prompt via tasks/*.format_prompt()
           b. Call llm_client.generate()
           c. Score via tasks/*.score_response()
           d. Optionally call llm_client.judge() with a rubric
        4. Compute fitness:
           task_score = mean(task_scores)
           if use_judge: fitness = (1 - judge_weight) * task_score + judge_weight * mean(judge_scores)
           else: fitness = task_score
    OUT: {fitness, task_score, judge_score, n_evaluated, sample_scores}
    """
    task_module = get_task_module(task_name)
    rubric = _RUBRICS[task_name]

    # 1. Load dataset
    dataset = task_module.load_task(data_path=data_path)

    # 2. Sample
    if len(dataset) <= n_samples:
        sample = dataset
    else:
        sample = random.sample(dataset, n_samples)

    task_scores: list[float] = []
    judge_scores: list[float] = []

    # 3. Evaluate each item
    for item in sample:
        formatted_question = task_module.format_prompt(item)

        # Generate
        try:
            response = llm_client.generate(
                model=model,
                system_prompt=prompt,
                user_prompt=formatted_question,
            )
        except OllamaError as e:
            logger.warning("OllamaError during generate for item %s: %s", item.get("id"), e)
            task_scores.append(0.0)
            if use_judge:
                judge_scores.append(0.0)
            continue

        # Task score
        try:
            t_score = _score_item(task_name, task_module, response, item)
        except Exception as e:
            logger.warning("Scoring error for item %s: %s", item.get("id"), e)
            t_score = 0.0
        task_scores.append(t_score)

        # Judge score (optional)
        if use_judge:
            try:
                j_score = llm_client.judge(
                    model=model,
                    question=formatted_question,
                    response=response,
                    rubric=rubric,
                )
            except OllamaError as e:
                logger.warning("OllamaError during judge for item %s: %s", item.get("id"), e)
                j_score = 0.0
            except Exception as e:
                logger.warning("Judge error for item %s: %s", item.get("id"), e)
                j_score = 0.0
            judge_scores.append(j_score)

    # 4. Aggregate
    n_evaluated = len(task_scores)
    task_score = sum(task_scores) / n_evaluated if n_evaluated > 0 else 0.0

    if use_judge and judge_scores:
        mean_judge = sum(judge_scores) / len(judge_scores)
        fitness = (1.0 - judge_weight) * task_score + judge_weight * mean_judge
    else:
        mean_judge = None
        fitness = task_score

    return {
        "fitness": fitness,
        "task_score": task_score,
        "judge_score": mean_judge,
        "n_evaluated": n_evaluated,
        "sample_scores": task_scores,
    }


def evaluate_prompt_dry_run(
    prompt: str,
    task_name: str,
    n_samples: int = 10,
) -> dict:
    """
    IN:  prompt, task_name, n_samples
    PROC: Return a fake result with random scores — no LLM call.
          Used for smoke-testing the pipeline without Ollama running.
    OUT: same shape as evaluate_prompt
    """
    # Validate task name early so callers catch typos
    get_task_module(task_name)

    sample_scores = [random.uniform(0.1, 0.9) for _ in range(n_samples)]
    task_score = sum(sample_scores) / len(sample_scores)

    return {
        "fitness": task_score,
        "task_score": task_score,
        "judge_score": None,
        "n_evaluated": n_samples,
        "sample_scores": sample_scores,
    }
