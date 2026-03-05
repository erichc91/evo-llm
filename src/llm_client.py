# llm_client.py — Thin wrapper around Ollama REST API
# All calls are synchronous (requests library).
# Raises OllamaError on connection failure so callers can handle gracefully.

import re
import requests

OLLAMA_BASE = "http://localhost:11434"


class OllamaError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """
    IN:  model name, system prompt, user prompt, temperature, max_tokens
    PROC: POST /api/chat with [system, user] messages
    OUT: response content string
    Raises OllamaError on connection failure or HTTP error.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }
    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(f"Ollama not reachable at {OLLAMA_BASE}: {e}") from e
    except requests.exceptions.HTTPError as e:
        raise OllamaError(f"Ollama HTTP error: {e}") from e

    return resp.json()["message"]["content"]


_JUDGE_TEMPLATE = """\
You are an impartial judge. Score the following response on a scale of 0-{max_score}.
Rubric: {rubric}
Question: {question}
Response: {response}
Reply with ONLY a number between 0 and {max_score}. No explanation.
Score:"""


def judge(
    model: str,
    question: str,
    response: str,
    rubric: str,
    max_score: int = 10,
) -> float:
    """
    IN:  model, original question, response to judge, rubric, max_score
    PROC: Build structured judge prompt; parse integer score from reply.
    OUT: float in [0.0, 1.0] (score / max_score). Returns 0.5 on parse failure.
    """
    prompt = _JUDGE_TEMPLATE.format(
        max_score=max_score,
        rubric=rubric,
        question=question,
        response=response,
    )
    try:
        raw = generate(model, system_prompt="", user_prompt=prompt, temperature=0.0, max_tokens=16)
    except OllamaError:
        return 0.5

    match = re.search(r"\b(\d+(?:\.\d+)?)\b", raw.strip())
    if not match:
        return 0.5

    try:
        score = float(match.group(1))
        score = max(0.0, min(float(max_score), score))
        return score / max_score
    except ValueError:
        return 0.5


def is_available(model: str) -> bool:
    """
    IN:  model name
    PROC: GET /api/tags, check if model present
    OUT: True/False (False on connection error)
    """
    models = list_models()
    return model in models


def list_models() -> list:
    """
    IN:  none
    PROC: GET /api/tags
    OUT: list of model name strings. Returns [] on error.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []
