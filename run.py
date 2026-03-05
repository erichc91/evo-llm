# run.py — CLI entry point for evo-llm evolutionary prompt optimizer
# Usage: python run.py [options]
# Example: python run.py --task reasoning --model phi3 --generations 10 --dry-run

import argparse
import sys

from src.fitness import get_seed_prompt
from src.evolve_prompt import init_population, score_population, run_generation, DEFAULT_CONFIG
from src.run_logger import init_run, log_generation, get_best, get_fitness_curve, list_runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sparkline(values: list) -> str:
    blocks = ' ▁▂▃▄▅▆▇█'
    if not values:
        return ''
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    return ''.join(blocks[int((v - mn) / rng * 8)] for v in values)


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_runs_table(runs: list) -> None:
    if not runs:
        print("No runs found.")
        return
    fmt = "{:<36}  {:<10}  {:<14}  {:>7}  {:>5}  {}"
    print(fmt.format("Run ID", "Task", "Model", "Best", "Gens", "Created"))
    print("-" * 90)
    for r in runs:
        created = r.get("created_at", "")[:19]  # trim microseconds
        print(fmt.format(
            r["run_id"],
            r.get("task", ""),
            r.get("model", ""),
            f"{r.get('best_fitness', 0.0):.4f}",
            r.get("generations_completed", 0),
            created,
        ))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="evo-llm: evolutionary prompt optimizer",
    )
    parser.add_argument("--task",          type=str,   default="reasoning",
                        metavar="TASK",
                        help="'reasoning' or 'coding'  (default: reasoning)")
    parser.add_argument("--model",         type=str,   default="phi3",
                        metavar="MODEL",
                        help="Ollama model name  (default: phi3)")
    parser.add_argument("--generations",   type=int,   default=20,
                        metavar="N",
                        help="Number of generations  (default: 20)")
    parser.add_argument("--pop-size",      type=int,   default=12,
                        metavar="N",
                        help="Population size  (default: 12)")
    parser.add_argument("--n-samples",     type=int,   default=10,
                        metavar="N",
                        help="Eval questions per organism  (default: 10)")
    parser.add_argument("--mutation-rate", type=float, default=0.5,
                        metavar="F",
                        help="Mutation probability  (default: 0.5)")
    parser.add_argument("--elite-count",   type=int,   default=2,
                        metavar="N",
                        help="Elites carried forward  (default: 2)")
    parser.add_argument("--seed-prompt",   type=str,   default=None,
                        metavar="STR",
                        help="Starting system prompt  (default: task's SEED_PROMPT)")
    parser.add_argument("--use-judge",     action="store_true",
                        help="Enable LLM-as-judge scoring")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Skip all LLM calls (test mode)")
    parser.add_argument("--resume",        type=str,   default=None,
                        metavar="RUN_ID",
                        help="Run ID to resume")
    parser.add_argument("--list-runs",     action="store_true",
                        help="List past runs and exit")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # 1. --list-runs: print table and exit
    if args.list_runs:
        _print_runs_table(list_runs())
        return

    # Build config dict (mirrors DEFAULT_CONFIG shape + run-level metadata)
    config = {
        "task":            args.task,
        "model":           args.model,
        "generations":     args.generations,
        "pop_size":        args.pop_size,
        "n_samples":       args.n_samples,
        "mutation_rate":   args.mutation_rate,
        "elite_count":     args.elite_count,
        "use_judge":       args.use_judge,
        "dry_run":         args.dry_run,
        "tournament_size": DEFAULT_CONFIG["tournament_size"],
    }

    # 2a. Print header
    dry_tag = "  [DRY-RUN]" if args.dry_run else ""
    print(f"\n{'━' * 50}")
    print(f"  evo-llm — evolutionary prompt optimizer{dry_tag}")
    print(f"{'━' * 50}")
    print(f"  Task:        {args.task}")
    print(f"  Model:       {args.model}")
    print(f"  Pop size:    {args.pop_size}")
    print(f"  Generations: {args.generations}")
    print(f"  N-samples:   {args.n_samples}")
    print(f"  Mut. rate:   {args.mutation_rate}")
    print(f"  Elites:      {args.elite_count}")
    print(f"  Use judge:   {args.use_judge}")
    print(f"{'━' * 50}\n")

    # 2b. Check Ollama availability (skip in dry-run)
    if not args.dry_run:
        from src.llm_client import is_available
        if not is_available(args.model):
            print(f"  WARNING: model '{args.model}' not found in Ollama.")
            print(f"  Ensure Ollama is running: ollama serve")
            print(f"  Pull the model with:      ollama pull {args.model}\n")

    # 2c. Seed prompt
    seed_prompt = args.seed_prompt if args.seed_prompt else get_seed_prompt(args.task)

    # 2d. Init run logger
    run_id = init_run(config)
    print(f"  Run ID: {run_id}\n")

    # 2e. Init population (no LLM diversification in dry-run)
    print("  Initializing population...", flush=True)
    population = init_population(
        seed_prompt,
        args.pop_size,
        args.model,
        diversify=not args.dry_run,
    )

    # 2f. Score generation 0
    print("  Scoring generation 0...", flush=True)
    population = score_population(
        population,
        task_name=args.task,
        model=args.model,
        n_samples=args.n_samples,
        use_judge=args.use_judge,
        dry_run=args.dry_run,
    )

    # 2g. Log generation 0
    log_generation(run_id, 0, population)

    fitnesses = [o["fitness"] for o in population]
    best_fit  = fitnesses[0]
    mean_fit  = _mean(fitnesses)
    preview   = population[0]["prompt"][:60].replace("\n", " ")
    print(f"  Gen  0/{args.generations} | Best: {best_fit:.4f} | Mean: {mean_fit:.4f} | \"{preview}\"")

    # 2h. Main evolutionary loop
    for gen in range(1, args.generations + 1):
        population = run_generation(
            population,
            task_name=args.task,
            model=args.model,
            config=config,
            dry_run=args.dry_run,
        )
        log_generation(run_id, gen, population)

        fitnesses = [o["fitness"] for o in population]
        best_fit  = fitnesses[0]
        mean_fit  = _mean(fitnesses)
        preview   = population[0]["prompt"][:60].replace("\n", " ")
        print(f"  Gen {gen:>2}/{args.generations} | Best: {best_fit:.4f} | Mean: {mean_fit:.4f} | \"{preview}\"")

    # 2i. Final summary
    best   = get_best(run_id)
    curve  = get_fitness_curve(run_id)
    bests  = [c["best_fitness"] for c in curve]
    spark  = sparkline(bests)
    first  = bests[0]  if bests else 0.0
    last   = bests[-1] if bests else 0.0

    print(f"\n{'━' * 50}")
    print(f"  Run complete!  ID: {run_id}")
    print(f"  Fitness: {spark}  ({first:.3f} → {last:.3f})")
    print(f"  Best prompt (gen {best.get('generation', '?')}, fitness {best['fitness']:.4f}):")
    print(f"    \"{best['prompt'][:80]}\"")
    print(f"{'━' * 50}\n")


if __name__ == "__main__":
    main()
