# evo-llm Quickstart

Evolutionary prompt optimizer — runs entirely locally on GPU via Ollama.

## Requirements
- Ollama installed + running (system tray or `ollama serve`)
- At least one model pulled

## Setup
```
cd C:\Users\Erich Curtis\Code_Projects\evo-llm
pip install -r requirements.txt
```

## Run
```
# Dry run (no LLM calls, test everything works)
python run.py --dry-run

# Run with fast model
python run.py --model llama3.2:3b --task reasoning --generations 10

# Run with quality model
python run.py --model phi3.5:latest --task coding --generations 20 --use-judge

# List past runs
python run.py --list-runs
```

## Recommended Models
| Model | Speed | Quality | Use |
|-------|-------|---------|-----|
| llama3.2:3b | Fast | Good | Development, quick runs |
| phi3.5:latest | Medium | High | Production runs |
| llama3.1:8b | Medium | High | Best balance |

Pull models: `ollama pull llama3.1:8b`

## Architecture
```
run.py
  -> evolve_prompt.py  (selection, mutation, crossover)
  -> fitness.py        (task definitions, scoring)
  -> llm_client.py     (Ollama REST API wrapper)
  -> run_logger.py     (SQLite run history in runs/)
```
