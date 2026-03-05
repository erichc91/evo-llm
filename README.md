# evo-llm

**Evolutionary prompt optimizer powered by a local LLM.**

Treats system prompts as genomes and evolves them via genetic algorithm — using the LLM itself as the mutation and crossover operator. Fitness is measured by actual task performance (reasoning accuracy or coding pass-rate), not vibes.

---

## Quick Start

**Prerequisites:**
1. Install [Ollama](https://ollama.com)
2. Pull a model: `ollama pull phi3.5`
3. Install dependencies: `pip install -r requirements.txt`

**Run the interactive UI (recommended):**
```bash
ollama serve          # start Ollama in background if not already running
python gradio_app.py  # opens browser at http://localhost:7860
```

**Run via CLI:**
```bash
python run.py --task reasoning --model phi3.5 --generations 20 --pop-size 12
```

**Smoke test (no LLM needed):**
```bash
python run.py --task reasoning --dry-run --generations 5
```

**List past runs:**
```bash
python run.py --list-runs
```

---

## How It Works

```
seed prompt --> diversify (LLM mutations) --> population[N]
                                                  |
                                          score on task dataset
                                          (N questions each)
                                                  |
                                     tournament select + breed
                                     (mutate or crossover via LLM)
                                                  |
                                          score new generation
                                                  |
                                          repeat x generations
                                                  |
                                       best prompt + fitness curve
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `reasoning` | `reasoning` or `coding` |
| `--model` | `phi3` | Any Ollama model name |
| `--generations` | `20` | Number of GA generations |
| `--pop-size` | `12` | Population size |
| `--n-samples` | `10` | Questions per fitness evaluation |
| `--mutation-rate` | `0.5` | Probability of mutate vs crossover |
| `--elite-count` | `2` | Elites carried forward unchanged |
| `--seed-prompt` | task default | Override starting prompt |
| `--use-judge` | off | Enable LLM-as-judge scoring |
| `--dry-run` | off | Skip all LLM calls (test mode) |
| `--list-runs` | — | Print past run history and exit |

---

## Project Structure

```
evo-llm/
  run.py                  # CLI entry point
  gradio_app.py           # Interactive Gradio UI (live fitness chart, prompt diffs)
  requirements.txt
  src/
    llm_client.py         # Ollama REST wrapper
    fitness.py            # Evaluator (task scoring + LLM judge)
    evolve_prompt.py      # Genetic algorithm (select/mutate/crossover)
    run_logger.py         # SQLite run history
    tasks/
      reasoning.py        # ARC-Easy exact-match scorer
      coding.py           # Python exec-based scorer
  data/
    arc_easy_sample.json  # 20 ARC-Easy questions
    coding_challenges.json
  docs/
    EXECUTIVE_SUMMARY.md
    TECHNICAL_DOCS.md
    ARCHITECTURE.md
```

---

## Adding a New Task

1. Copy `src/tasks/reasoning.py` → `src/tasks/mytask.py`
2. Implement:
   - `SEED_PROMPT` — starting system prompt
   - `load_task(data_path=None)` → `list[dict]`
   - `format_prompt(item)` → `str`
   - `score_response(response, item)` → `float` in `[0.0, 1.0]`
3. Add data file to `data/`
4. Run: `python run.py --task mytask`

---

## Roadmap

| Tier | Genome | Status |
|------|--------|--------|
| 1 — Prompt evolution | Text string | **Done** |
| 2 — Merge coefficient evolution | Float[] per-layer weights | Planned |
| 3 — LoRA adapter evolution | Adapter weights delta | Planned |

---

## Docs

- [Executive Summary](docs/EXECUTIVE_SUMMARY.md) — plain English overview
- [Technical Docs](docs/TECHNICAL_DOCS.md) — architecture, modules, gotchas
- [Architecture Diagram](docs/ARCHITECTURE.md) — ASCII system diagrams
