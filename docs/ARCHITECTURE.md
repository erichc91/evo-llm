# evo-llm — Architecture Diagram

**Last Updated:** 2026-03-05 (Session 1)

---

## System Overview

```
+------------------+      +-----------------------+      +------------------+
|   run.py (CLI)   |      |   evolve_prompt.py    |      |  run_logger.py   |
|                  |      |   Genetic Algorithm   |      |  SQLite History  |
|  --task          +----->|                       +----->|                  |
|  --model         |      |  init_population()    |      |  runs/{id}.db    |
|  --generations   |      |  score_population()   |      |  run_config      |
|  --pop-size      |      |  run_generation()     |      |  generations     |
|  --n-samples     |      |  tournament_select()  |      |  organisms       |
|  --dry-run       |      |  llm_mutate()         |      |                  |
+------------------+      |  llm_crossover()      |      +------------------+
                          +-----------+-----------+
                                      |
                    +-----------------+-----------------+
                    |                                   |
         +----------v-----------+          +-----------v----------+
         |    fitness.py        |          |   llm_client.py      |
         |  Prompt Evaluator    |          |   Ollama REST Wrap   |
         |                      |          |                      |
         |  evaluate_prompt()   |          |  generate()          |
         |  load_task()         |          |  judge()             |
         |  score per sample    |          |  is_available()      |
         +----------+-----------+          +-----------+----------+
                    |                                  |
         +----------v-----------+          +-----------v----------+
         |   tasks/             |          |  Ollama Server       |
         |  reasoning.py        |          |  localhost:11434     |
         |  coding.py           |          |                      |
         |                      |          |  phi3.5 (local LLM)  |
         |  load_task()         |          |  2.2 GB, ~300 tok/s  |
         |  format_prompt()     |          |                      |
         |  score_response()    |          +----------------------+
         +----------+-----------+
                    |
         +----------v-----------+
         |   data/              |
         |  arc_easy_sample.json|
         |  coding_challenges   |
         |  .json               |
         +----------------------+
```

---

## Data Flow Detail

```
INPUTS                    PROCESSING                      OUTPUTS
------                    ----------                      -------

seed_prompt               init_population()
(text string)  +--------> diversify via llm_mutate()  --> population[12]
                          (or seed copies if dry-run)      list of organisms


population[N]             score_population()
(unscored)     +--------> for each organism:           --> population[N]
                          - sample N questions              sorted best-first
                          - call generate()                 fitness = [0.0, 1.0]
                          - score response
                          - optionally judge()


population[N]             run_generation()
(scored, gen K)+--------> 1. keep top 2 (elites)       --> population[N]
                          2. breed N-2 children              gen K+1, re-scored
                          3. mutate or crossover
                          4. evaluate children
                          5. sort by fitness


population[N]             log_generation()
(gen K+1)      +--------> INSERT into generations      --> runs/{uuid}.db
                               and organisms tables          fitness history


runs/{uuid}.db            sparkline()
fitness curve  +--------> build unicode chart          --> terminal output
                                                            "▁▂▃▄▅▆▇█"
```

---

## Genetic Algorithm Loop

```
                  +---------------------------+
                  |       GENERATION 0        |
                  |                           |
  seed_prompt --> |  [P1] [P2] ... [P12]      |
                  |   diversify via LLM       |
                  +----------+----------------+
                             |
                    score all organisms
                    (N questions each)
                             |
                  +----------v----------------+
                  |     SELECTION             |
                  |                           |
                  |  elite_count=2  --------> carry forward unchanged
                  |  10 remaining  --------> tournament_select() x2
                  +----------+----------------+
                             |
              +--------------+-------------+
              |                            |
     random() < 0.5                 random() >= 0.5
              |                            |
     llm_mutate(parent)          llm_crossover(A, B)
     rephrase / extend / trim     blend best elements
              |                            |
              +-------------+--------------+
                            |
                    score new children
                            |
                  +---------v-----------------+
                  |     GENERATION 1          |
                  |  sorted best-first        |
                  |  repeat for N gens        |
                  +---------------------------+
```

---

## Mutation & Crossover Operators (LLM-powered)

```
MUTATION TYPES
--------------

rephrase:   [prompt] --> LLM: "Rewrite more clearly"   --> [cleaner prompt]
extend:     [prompt] --> LLM: "Add one instruction"    --> [more specific prompt]
trim:       [prompt] --> LLM: "Remove least important" --> [shorter prompt]

                        On OllamaError: return original prompt unchanged


CROSSOVER
---------

[prompt_A]  )
            +--> LLM: "Combine best elements" --> [hybrid prompt]
[prompt_B]  )

                        On OllamaError: return prompt_A unchanged
```

---

## Run Storage Layout

```
runs/
  {uuid-1}.db   <-- Run 1: reasoning task, phi3.5, 20 gens
  {uuid-2}.db   <-- Run 2: coding task, phi3.5, 10 gens
  {uuid-3}.db   <-- Run 3: ...

python run.py --list-runs
  --> reads all .db files
  --> prints table: run_id | task | model | best | gens | created
```

---

## Tier Roadmap

```
TIER 1 (DONE)           TIER 2 (PLANNED)         TIER 3 (PLANNED)
--------------          ----------------         ----------------
Prompt evolution        Merge coefficient        LoRA adapter
                        evolution                evolution
Genome = text string    Genome = float[]         Genome = adapter weights
                        (per-layer merge         (trained delta)
Mutation = LLM          coefficients)
rewrite                 Mutation = gaussian      Mutation = gradient
                        noise on coeffs          perturbation
Fitness = task          Fitness = task           Fitness = task
accuracy                accuracy                 accuracy
Model = unchanged       Model = merged           Model = fine-tuned
(prompt only)           (mergekit-evolve)        (peft/LoRA)
Hardware: CPU/RAM       Hardware: 8+ GB VRAM     Hardware: 12+ GB VRAM
Time: mins              Time: hours              Time: days
```
