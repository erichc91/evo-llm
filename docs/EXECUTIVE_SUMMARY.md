# evo-llm — Executive Summary

**Owner:** Erich Curtis
**Created:** 2026-03-05
**Last Updated:** 2026-03-05 (Session 1)
**Status:** In Progress

---

## What This Is

evo-llm is an evolutionary AI system that automatically improves a language model's
system prompt over time — using the LLM itself as the mutation and crossover engine.
You give it a task (like "answer science questions"), and it breeds better and better
prompts across generations, keeping only the ones that actually score higher.

---

## What It Does For You Today

### On demand — via command line
Run an evolutionary experiment and watch a fitness curve climb over ~10–20 minutes:
- Starts from a seed system prompt
- Breeds a population of 8–12 prompt variants using phi3.5 (local, free, private)
- Scores each prompt by actually running it against real test questions
- Keeps the best, breeds new ones, repeats for N generations
- Saves every run to a SQLite database so you can compare across experiments
- Outputs a sparkline progress chart in the terminal

### What you get at the end
The best system prompt discovered, with its fitness score — something a human
would take hours to write and tune by hand. The system finds prompt patterns
that genuinely improve model accuracy.

---

## What It Does NOT Do (Yet)

| Feature | Effort | Status |
|---------|--------|--------|
| Fitness curve plot (matplotlib chart) | 30 min | Pending |
| Prompt diff view (gen-to-gen changes) | 1 hr | Pending |
| Baseline comparison (zero-shot vs evolved) | 1 hr | Pending |
| Tier 2: Merge coefficient evolution (per-layer weights) | 1–2 days | Planned |
| Tier 3: LoRA adapter evolution | 3–5 days | Planned |
| Resume interrupted run (`--resume RUN_ID`) | 2 hrs | Built, untested |
| Web UI for results explorer | 2–4 hrs | Backlog |

---

## How to Maintain It

**Prerequisites (one-time setup):**
- Install Ollama: https://ollama.com
- Pull a model: `ollama pull phi3.5`
- Ollama server starts automatically with Windows

**Run an experiment:**
```
cd evo-llm
python run.py --task reasoning --model phi3.5 --generations 20 --pop-size 12
```

**Quick smoke-test (no LLM needed):**
```
python run.py --task reasoning --dry-run --generations 5
```

**List past runs:**
```
python run.py --list-runs
```

**Add a new task type:**
- Copy `src/tasks/reasoning.py` to `src/tasks/yourname.py`
- Implement `load_task()`, `format_prompt()`, `score_response()`, `SEED_PROMPT`
- Add data file to `data/`

**If Ollama is not running:**
```
ollama serve
```
(or restart your computer — it auto-starts on Windows after install)
