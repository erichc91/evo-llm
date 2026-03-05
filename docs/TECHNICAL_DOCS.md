# evo-llm — Technical Documentation

**Last Updated:** 2026-03-05 (Session 1)
**Python:** 3.11 | **Venv:** `evo-llm\venv\` (create with `python -m venv venv`)
**Dependencies:** `requests>=2.31.0` only

---

## Project Structure

```
evo-llm/
  run.py                      # CLI entry point (argparse)
  requirements.txt            # requests only
  docs/
    EXECUTIVE_SUMMARY.md
    TECHNICAL_DOCS.md
    ARCHITECTURE.md
  src/
    __init__.py
    llm_client.py             # Ollama REST wrapper
    fitness.py                # Prompt evaluator + dataset loader
    evolve_prompt.py          # Genetic algorithm core
    run_logger.py             # SQLite run history
    tasks/
      __init__.py
      reasoning.py            # ARC-Easy exact-match scorer
      coding.py               # Python exec-based scorer
  data/
    arc_easy_sample.json      # 20 ARC-Easy reasoning questions
    coding_challenges.json    # 5 Python challenges with test cases
  runs/                       # Auto-created; one .db file per run (git-ignored)
```

---

## External Dependencies

| Dependency | Version | Purpose | Auth |
|-----------|---------|---------|------|
| Ollama | Local server | LLM inference backend | None — local HTTP |
| phi3.5 (or any model) | Pulled via `ollama pull` | Mutation, crossover, inference, judging | None |
| requests | >=2.31.0 | HTTP calls to Ollama REST API | None |
| sqlite3 | stdlib | Run history logging | None |

All inference is local. No cloud APIs, no keys, no data leaves the machine.

---

## Organism Data Format

```python
organism = {
    "prompt":     str,    # The system prompt string (the "genome")
    "fitness":    float,  # Score in [0.0, 1.0]; higher = better
    "generation": int,    # Which generation this organism was born in
    "parent_ids": list,   # Indices of parents in previous generation
}
```

Population = `list[organism]`, always sorted best-first after scoring.

---

## llm_client.py — Ollama REST Wrapper

```
generate(model, system_prompt, user_prompt)
    --> POST /api/chat  (stream=False)
    <-- response content string
    raises OllamaError on connection/HTTP failure

judge(model, question, response, rubric, max_score=10)
    --> generate() with structured judge template
    <-- float in [0.0, 1.0]  (returns 0.5 on parse failure)

is_available(model) --> bool
list_models()       --> list[str]
```

- Base URL: `http://localhost:11434`
- Timeout: 120s for generate, 10s for list_models
- Temperature: 0.7 default for generate, 0.9 for mutation, 0.8 for crossover, 0.0 for judge
- OllamaError is the only exception raised; all callers catch it and degrade gracefully

---

## fitness.py — Prompt Evaluator

```
evaluate_prompt(prompt, task_name, model, n_samples, use_judge)
    1. load_task()          -- load JSON dataset
    2. random.sample()      -- pick n_samples items
    3. format_prompt(item)  -- build user question string
    4. llm_client.generate()
    5. score_response()     -- task-specific scorer
    6. optionally: llm_client.judge() weighted by judge_weight=0.3
    <-- {fitness, task_score, judge_score, n_evaluated, sample_scores}

evaluate_prompt_dry_run(prompt, task_name, n_samples)
    --> returns random scores in [0.1, 0.9], no LLM calls
```

### Fitness formula
- Default (no judge): `fitness = mean(task_scores)`
- With judge: `fitness = 0.7 * task_score + 0.3 * mean(judge_scores)`

### Task score contracts
- `reasoning.score_response(response, answer_str)` — string exact match, returns 1.0 or 0.0
- `coding.score_response(response, item_dict)` — executes code, runs test cases, returns pass-rate

---

## evolve_prompt.py — Genetic Algorithm

```
init_population(seed_prompt, pop_size, model, diversify=True)
    --> pop_size-1 llm_mutate() calls on seed; fills with seed copies on failure

score_population(population, task_name, model, n_samples, use_judge, dry_run)
    --> evaluate_prompt() on every organism; sort desc by fitness; return

run_generation(population, task_name, model, config, dry_run)
    1. Sort current population
    2. Carry elite_count organisms forward unchanged
    3. For each remaining slot:
       - random() < mutation_rate  -->  llm_mutate(tournament_winner.prompt)
       - else                      -->  llm_crossover(parent_a.prompt, parent_b.prompt)
    4. Evaluate all new organisms
    5. Sort and return next generation
```

### Mutation types (chosen randomly)
| Type | Instruction to LLM |
|------|--------------------|
| `rephrase` | Rewrite more clearly, preserve all instructions |
| `extend` | Add one specific, actionable instruction |
| `trim` | Remove the least important sentence |

### DEFAULT_CONFIG
```python
{
    "pop_size": 12,
    "n_samples": 10,
    "use_judge": False,
    "mutation_rate": 0.5,    # prob of mutate vs crossover per child
    "tournament_size": 3,
    "elite_count": 2,
}
```

---

## run_logger.py — SQLite Run History

DB file: `runs/{run_id}.db` (UUID per run, auto-created)

### Tables

```sql
-- One row per run
CREATE TABLE run_config (
    run_id TEXT PRIMARY KEY,
    task TEXT, model TEXT, pop_size INTEGER,
    generations INTEGER, n_samples INTEGER,
    mutation_rate REAL, elite_count INTEGER, use_judge INTEGER,
    created_at TEXT
);

-- One row per generation per run
CREATE TABLE generations (
    run_id TEXT, generation INTEGER,
    best_fitness REAL, mean_fitness REAL,
    best_prompt TEXT, logged_at TEXT,
    PRIMARY KEY (run_id, generation)
);

-- One row per organism per generation
CREATE TABLE organisms (
    run_id TEXT, generation INTEGER, rank INTEGER,
    prompt TEXT, fitness REAL, parent_ids TEXT,
    PRIMARY KEY (run_id, generation, rank)
);
```

### Key functions
- `init_run(config) -> run_id`
- `log_generation(run_id, gen_num, population)`
- `get_best(run_id) -> organism dict`
- `get_fitness_curve(run_id) -> list[{generation, best_fitness, mean_fitness}]`
- `list_runs() -> list of run_config rows` (cross-run, reads all .db files in runs/)

---

## src/tasks/reasoning.py

- Dataset: `data/arc_easy_sample.json` (20 ARC-Easy 4-choice questions)
- SEED_PROMPT: instructs model to answer multiple-choice, respond with letter only
- `format_prompt(item)` → "Question: ... \nChoices: A) ... B) ... C) ... D) ..."
- `score_response(response, answer)` → 1.0 if answer letter in response, else 0.0

## src/tasks/coding.py

- Dataset: `data/coding_challenges.json` (5 Python function challenges)
- SEED_PROMPT: instructs model to write a Python function, no extra text
- `format_prompt(item)` → "Write a Python function: ... \nSignature: ..."
- `score_response(response, item)` → extracts code block, exec()s it, runs test cases, returns pass-rate [0.0, 1.0]

---

## Known Issues & Gotchas

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `ollama list` stderr warning "failed to get console mode" | Windows console handle quirk | Harmless; ignored |
| `phi3.5` model not found warning on first run | `is_available()` checks exact name; Ollama stores as `phi3.5:latest` | Warning only; generate() still works if model is pulled |
| Dry-run mode: mutation/crossover still attempt Ollama | By design — tests graceful fallback path | Correct behavior; falls back to original prompt on OllamaError |
| `runs/` dir must exist | Created by `init_run()` on first call | Auto-created; no manual step needed |
| Coding scorer: `exec()` in subprocess-less environment | Security risk if running user-supplied code | Acceptable for local use; do not expose as a web service |

---

## Tier 2 / Tier 3 Backlog

| Feature | File to Create | Key Dependency |
|---------|---------------|----------------|
| Fitness curve plot | `src/visualize.py` | matplotlib |
| Prompt diff viewer | `src/diff_view.py` | difflib (stdlib) |
| Baseline comparator | `src/baseline.py` | run_logger |
| Merge coefficient evolution (Tier 2) | `src/merge_evolve.py` | mergekit, two compatible base models |
| LoRA adapter evolution (Tier 3) | `src/lora_evolve.py` | peft, transformers, CUDA GPU |
| Resume run from checkpoint | Already wired in run.py (`--resume`) | run_logger.get_population_at_gen() — not yet implemented |
| Web results explorer | `src/web_ui.py` | flask or streamlit |
