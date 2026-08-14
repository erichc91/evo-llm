# evolve_prompt.py — Genetic Algorithm over system prompt strings
# Organisms = text strings (system prompts)
# Mutation/crossover = LLM rewrites
# Selection = tournament selection (fitness-proportional)

import logging
import random

from src import llm_client
from src.llm_client import OllamaError
from src.fitness import evaluate_prompt, evaluate_prompt_dry_run

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "pop_size": 12,
    "n_samples": 10,
    "use_judge": False,
    "mutation_rate": 0.5,   # prob of mutate vs crossover for non-elites
    "tournament_size": 3,
    "elite_count": 2,
}

# ---------------------------------------------------------------------------
# Mutation meta-prompts
# ---------------------------------------------------------------------------

_MUTATION_STYLES = {
    "rephrase": "Rewrite this system prompt more clearly, preserving all instructions",
    "extend":   "Add one specific, actionable instruction to improve this system prompt",
    "trim":     "Remove the least important sentence from this system prompt",
}

_MUTATION_TEMPLATE = """\
You are a prompt engineer. Your task: {mutation_instruction}

Original prompt:
{prompt}

Output ONLY the new system prompt. No explanation, no quotes, no markdown.
New prompt:"""

_CROSSOVER_TEMPLATE = """\
You are a prompt engineer. Combine the best elements of these two system prompts into one improved prompt.

Prompt A:
{parent_a}

Prompt B:
{parent_b}

Output ONLY the combined system prompt. No explanation, no quotes, no markdown.
Combined prompt:"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_organism(prompt: str, generation: int = 0, parent_ids: list = None) -> dict:
    """Build a fresh Organism dict with zero fitness."""
    return {
        "prompt":     prompt,
        "fitness":    0.0,
        "generation": generation,
        "parent_ids": parent_ids if parent_ids is not None else [],
    }


def _eval(
    prompt: str,
    task_name: str,
    model: str,
    n_samples: int,
    use_judge: bool,
    dry_run: bool,
) -> dict:
    """Route to real or dry-run evaluator."""
    if dry_run:
        return evaluate_prompt_dry_run(prompt, task_name, n_samples)
    return evaluate_prompt(prompt, task_name, model, n_samples, use_judge)


def _find_index(population: list, organism: dict) -> int:
    """Return index of organism by identity (for parent_ids logging)."""
    for i, o in enumerate(population):
        if o is organism:
            return i
    return -1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_population(
    seed_prompt: str,
    pop_size: int,
    model: str,
    diversify: bool = True,
) -> list:
    """
    IN:  seed_prompt (str), pop_size (int), model (str), diversify (bool)
    PROC:
        - First organism = seed_prompt as-is
        - If diversify=True: generate pop_size-1 variations via llm_mutate
        - If diversify=False or LLM fails: fill remaining slots with seed copies
        - All organisms: fitness=0.0, generation=0, parent_ids=[]
    OUT: list of Organism dicts
    """
    population = [_make_organism(seed_prompt, generation=0)]

    for _ in range(pop_size - 1):
        if diversify:
            variant = llm_mutate(seed_prompt, model)
        else:
            variant = seed_prompt
        population.append(_make_organism(variant, generation=0))

    return population


def tournament_select(scored_pop: list, tournament_size: int = 3) -> dict:
    """
    IN:  scored_pop (list of Organisms with fitness set), tournament_size (int)
    PROC: randomly pick tournament_size organisms; return highest-fitness one
    OUT: one Organism (winner)
    """
    contestants = random.sample(scored_pop, min(tournament_size, len(scored_pop)))
    return max(contestants, key=lambda o: o["fitness"])


def _offline_mutate(prompt: str, mutation_type: str) -> str:
    """Offline stand-in for llm_mutate, used when dry_run=True.

    Deliberately returns a *different* string rather than the original: if
    every organism came back identical, dry-run would exercise the plumbing
    but not the search, and a broken selection step would still look fine.
    """
    words = prompt.split()
    if mutation_type == "trim" and len(words) > 4:
        return " ".join(words[: max(3, int(len(words) * 0.8))])
    if mutation_type == "extend":
        return prompt.rstrip() + " Explain your reasoning step by step."
    if len(words) > 3:
        i = random.randrange(0, len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def _offline_crossover(parent_a: str, parent_b: str) -> str:
    """Offline stand-in for llm_crossover, used when dry_run=True."""
    a, b = parent_a.split(), parent_b.split()
    merged = a[: max(1, len(a) // 2)] + b[len(b) // 2 :]
    return " ".join(merged) or parent_a


def llm_mutate(prompt: str, model: str, mutation_type: str = None,
               dry_run: bool = False) -> str:
    """
    IN:  prompt (str), model (str), mutation_type (str or None), dry_run (bool)
    PROC:
        - Randomly choose from rephrase / extend / trim if mutation_type is None
        - If dry_run: return an offline synthetic mutation, making no LLM call
        - Otherwise build meta-prompt and call llm_client.generate()
        - On OllamaError: return original prompt unchanged
    OUT: new (or original on failure) prompt string
    """
    if mutation_type is None:
        mutation_type = random.choice(list(_MUTATION_STYLES.keys()))

    if dry_run:
        return _offline_mutate(prompt, mutation_type)

    instruction = _MUTATION_STYLES.get(mutation_type, _MUTATION_STYLES["rephrase"])
    meta_prompt = _MUTATION_TEMPLATE.format(
        mutation_instruction=instruction,
        prompt=prompt,
    )

    try:
        result = llm_client.generate(
            model=model,
            system_prompt="",
            user_prompt=meta_prompt,
            temperature=0.9,
            max_tokens=512,
        )
        return result.strip() or prompt
    except OllamaError as e:
        logger.warning("llm_mutate failed (%s): %s — returning original", mutation_type, e)
        return prompt


def llm_crossover(parent_a: str, parent_b: str, model: str,
                  dry_run: bool = False) -> str:
    """
    IN:  parent_a (str), parent_b (str), model (str), dry_run (bool)
    PROC:
        - If dry_run: splice the two parents offline, making no LLM call
        - Otherwise call LLM to merge best elements of both prompts
        - On OllamaError: return parent_a unchanged
    OUT: new combined prompt string (or parent_a on failure)
    """
    if dry_run:
        return _offline_crossover(parent_a, parent_b)

    meta_prompt = _CROSSOVER_TEMPLATE.format(parent_a=parent_a, parent_b=parent_b)

    try:
        result = llm_client.generate(
            model=model,
            system_prompt="",
            user_prompt=meta_prompt,
            temperature=0.8,
            max_tokens=512,
        )
        return result.strip() or parent_a
    except OllamaError as e:
        logger.warning("llm_crossover failed: %s — returning parent_a", e)
        return parent_a


def score_population(
    population: list,
    task_name: str,
    model: str,
    n_samples: int,
    use_judge: bool = False,
    dry_run: bool = False,
) -> list:
    """
    IN:  unevaluated population, task/model params, dry_run flag
    PROC: evaluate each organism; set .fitness in-place; return sorted desc
    OUT: same list with fitness set, sorted best-first
    """
    for organism in population:
        result = _eval(organism["prompt"], task_name, model, n_samples, use_judge, dry_run)
        organism["fitness"] = result["fitness"]

    population.sort(key=lambda o: o["fitness"], reverse=True)
    return population


def run_generation(
    population: list,
    task_name: str,
    model: str,
    config: dict,
    dry_run: bool = False,
) -> list:
    """
    IN:
        population  — current generation (already scored)
        task_name   — str
        model       — str
        config      — {pop_size, n_samples, use_judge, mutation_rate, tournament_size, elite_count}
        dry_run     — if True, use evaluate_prompt_dry_run instead of real eval
    PROC:
        1. Sort population by fitness desc
        2. Carry top elite_count organisms unchanged into next gen
        3. Fill remaining slots: mutate (prob=mutation_rate) or crossover two tournament winners
        4. Evaluate all new (non-elite) organisms
        5. Set fitness + generation number on each new organism
    OUT: new generation as list[Organism], sorted by fitness desc
    """
    pop_size        = config.get("pop_size",        DEFAULT_CONFIG["pop_size"])
    n_samples       = config.get("n_samples",        DEFAULT_CONFIG["n_samples"])
    use_judge       = config.get("use_judge",         DEFAULT_CONFIG["use_judge"])
    mutation_rate   = config.get("mutation_rate",    DEFAULT_CONFIG["mutation_rate"])
    tournament_size = config.get("tournament_size",  DEFAULT_CONFIG["tournament_size"])
    elite_count     = config.get("elite_count",      DEFAULT_CONFIG["elite_count"])

    next_gen_num = (max(o["generation"] for o in population) if population else 0) + 1

    # 1. Sort current population best-first
    population.sort(key=lambda o: o["fitness"], reverse=True)

    # 2. Elites carry over as shallow copies
    next_gen = [dict(o) for o in population[:elite_count]]

    # 3. Breed new organisms to fill remaining slots
    new_organisms = []
    for _ in range(pop_size - len(next_gen)):
        if random.random() < mutation_rate:
            parent = tournament_select(population, tournament_size)
            new_prompt = llm_mutate(parent["prompt"], model, dry_run=dry_run)
            child = _make_organism(
                new_prompt,
                generation=next_gen_num,
                parent_ids=[_find_index(population, parent)],
            )
        else:
            parent_a = tournament_select(population, tournament_size)
            parent_b = tournament_select(population, tournament_size)
            new_prompt = llm_crossover(parent_a["prompt"], parent_b["prompt"], model,
                                       dry_run=dry_run)
            child = _make_organism(
                new_prompt,
                generation=next_gen_num,
                parent_ids=[
                    _find_index(population, parent_a),
                    _find_index(population, parent_b),
                ],
            )
        new_organisms.append(child)

    # 4. Evaluate new organisms (elites skip re-evaluation)
    for child in new_organisms:
        result = _eval(child["prompt"], task_name, model, n_samples, use_judge, dry_run)
        child["fitness"] = result["fitness"]

    next_gen.extend(new_organisms)

    # 5. Sort and return
    next_gen.sort(key=lambda o: o["fitness"], reverse=True)
    return next_gen
