# gradio_app.py — Interactive UI for evo-llm
# Wraps existing backend (evolve_prompt.py / fitness.py) with zero changes.
# Run: python gradio_app.py

import threading
import difflib
import time
import sys
import os

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import plotly.graph_objects as go

from src.fitness import get_seed_prompt
from src.evolve_prompt import (
    init_population,
    score_population,
    run_generation,
    DEFAULT_CONFIG,
)
from src.llm_client import list_models

# ---------------------------------------------------------------------------
# Globals for the running experiment (single-threaded Gradio event loop is fine)
# ---------------------------------------------------------------------------

_state = {
    "running":    False,
    "stop":       False,
    "population": [],
    "history":    [],   # [{gen, best, mean, best_prompt, all_prompts}]
    "log":        [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _diff_highlight(old: str, new: str):
    """
    Return a list of (text, label) tuples for gr.HighlightedText.
    Unchanged = None, added = "added", removed = "removed"
    """
    matcher = difflib.SequenceMatcher(None, old.split(), new.split())
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            result.append((" ".join(old.split()[i1:i2]), None))
        elif tag == "replace":
            result.append((" ".join(old.split()[i1:i2]), "removed"))
            result.append((" ".join(new.split()[j1:j2]), "added"))
        elif tag == "delete":
            result.append((" ".join(old.split()[i1:i2]), "removed"))
        elif tag == "insert":
            result.append((" ".join(new.split()[j1:j2]), "added"))
    # Flatten — add spaces between words
    flat = []
    for text, label in result:
        for word in text.split():
            flat.append((word + " ", label))
    return flat if flat else [(new, None)]


def _fitness_chart(history):
    if not history:
        fig = go.Figure()
        fig.update_layout(
            title="Fitness over Generations",
            xaxis_title="Generation",
            yaxis_title="Fitness",
            yaxis_range=[0, 1],
            template="plotly_dark",
            height=300,
        )
        return fig

    gens  = [h["gen"] for h in history]
    bests = [h["best"] for h in history]
    means = [h["mean"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens, y=bests, mode="lines+markers",
        name="Best", line=dict(color="#00ff88", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=means, mode="lines",
        name="Mean", line=dict(color="#6699ff", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title="Fitness over Generations",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        yaxis_range=[0, 1],
        template="plotly_dark",
        height=300,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def _population_table(population):
    if not population:
        return []
    rows = []
    for i, org in enumerate(population):
        rank = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"  {i+1}"))
        fitness_bar = "█" * int(org["fitness"] * 10) + "░" * (10 - int(org["fitness"] * 10))
        preview = org["prompt"][:80].replace("\n", " ") + ("…" if len(org["prompt"]) > 80 else "")
        rows.append([rank, f"{org['fitness']:.4f}  {fitness_bar}", f"Gen {org['generation']}", preview])
    return rows


def _available_models():
    models = list_models()
    return models if models else ["phi3.5", "phi3", "llama3", "mistral"]


# ---------------------------------------------------------------------------
# Core experiment runner (runs in background thread)
# ---------------------------------------------------------------------------

def _run_experiment(task, model, generations, pop_size, n_samples, mutation_rate, elite_count, seed_prompt_override, use_judge):
    _state["running"] = True
    _state["stop"]    = False
    _state["history"] = []
    _state["log"]     = []
    _state["population"] = []

    def log(msg):
        _state["log"].append(msg)

    config = {
        "pop_size":        pop_size,
        "n_samples":       n_samples,
        "mutation_rate":   mutation_rate,
        "elite_count":     elite_count,
        "use_judge":       use_judge,
        "tournament_size": DEFAULT_CONFIG["tournament_size"],
    }

    seed = seed_prompt_override.strip() if seed_prompt_override.strip() else get_seed_prompt(task)
    log(f"Seed prompt: {seed[:80]}...")
    log(f"Initializing population (size={pop_size})...")

    population = init_population(seed, pop_size, model, diversify=True)

    log("Scoring generation 0...")
    population = score_population(population, task, model, n_samples, use_judge, dry_run=False)

    fitnesses = [o["fitness"] for o in population]
    _state["history"].append({
        "gen": 0,
        "best": fitnesses[0],
        "mean": _mean(fitnesses),
        "best_prompt": population[0]["prompt"],
        "all_prompts": [o["prompt"] for o in population],
    })
    _state["population"] = population
    log(f"Gen 0 — Best: {fitnesses[0]:.4f}  Mean: {_mean(fitnesses):.4f}")

    for gen in range(1, generations + 1):
        if _state["stop"]:
            log("Stopped by user.")
            break

        log(f"Running generation {gen}/{generations}...")
        population = run_generation(population, task, model, config, dry_run=False)

        fitnesses = [o["fitness"] for o in population]
        prev_best = _state["history"][-1]["best_prompt"]
        _state["history"].append({
            "gen": gen,
            "best": fitnesses[0],
            "mean": _mean(fitnesses),
            "best_prompt": population[0]["prompt"],
            "all_prompts": [o["prompt"] for o in population],
        })
        _state["population"] = population
        log(f"Gen {gen} — Best: {fitnesses[0]:.4f}  Mean: {_mean(fitnesses):.4f}")

    log("Done!" if not _state["stop"] else "Stopped.")
    _state["running"] = False


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def start_run(task, model, generations, pop_size, n_samples, mutation_rate, elite_count, seed_override, use_judge):
    if _state["running"]:
        return "Already running — stop first."
    t = threading.Thread(
        target=_run_experiment,
        args=(task, model, int(generations), int(pop_size), int(n_samples),
              float(mutation_rate), int(elite_count), seed_override, use_judge),
        daemon=True,
    )
    t.start()
    return "Started!"


def stop_run():
    _state["stop"] = True
    return "Stop signal sent."


def poll_ui():
    """Called every 3 seconds by gr.Timer to refresh all outputs."""
    history    = _state["history"]
    population = _state["population"]
    log_lines  = _state["log"]

    chart      = _fitness_chart(history)
    pop_table  = _population_table(population)
    log_text   = "\n".join(log_lines[-30:])  # last 30 lines

    # Best prompt diff (last two gens)
    if len(history) >= 2:
        prev = history[-2]["best_prompt"]
        curr = history[-1]["best_prompt"]
        diff = _diff_highlight(prev, curr)
    elif len(history) == 1:
        diff = [(w + " ", None) for w in history[0]["best_prompt"].split()]
    else:
        diff = [("No data yet.", None)]

    # Full best prompt text
    best_text = history[-1]["best_prompt"] if history else ""

    # Status badge
    if _state["running"]:
        gen_num = history[-1]["gen"] if history else 0
        status  = f"⚙️  Running — Gen {gen_num}"
    elif history:
        status = f"✅  Done — {len(history)-1} generations"
    else:
        status = "⏸  Idle"

    return chart, pop_table, diff, best_text, log_text, status


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_ui():
    models = _available_models()

    with gr.Blocks(title="evo-llm") as app:

        gr.Markdown("# 🧬 evo-llm — Evolutionary Prompt Optimizer")
        gr.Markdown("Evolves system prompts via genetic algorithm using a local LLM. Watch fitness improve in real time.")

        with gr.Row():
            # --- Left: controls ---
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Parameters")
                task          = gr.Dropdown(["reasoning", "coding"], value="reasoning", label="Task")
                model         = gr.Dropdown(models, value=models[0] if models else "phi3.5", label="Model")
                generations   = gr.Slider(1, 50, value=10, step=1, label="Generations")
                pop_size      = gr.Slider(2, 20, value=6,  step=1, label="Population Size")
                n_samples     = gr.Slider(1, 20, value=5,  step=1, label="Questions per Eval")
                mutation_rate = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Mutation Rate")
                elite_count   = gr.Slider(0, 5, value=2, step=1, label="Elites Carried Forward")
                use_judge     = gr.Checkbox(label="Use LLM-as-Judge scoring", value=False)
                seed_override = gr.Textbox(
                    label="Seed Prompt (editable — this is where evolution starts)",
                    lines=4,
                    value=get_seed_prompt("reasoning"),
                )
                with gr.Row():
                    btn_start = gr.Button("▶  Start", variant="primary")
                    btn_stop  = gr.Button("⏹  Stop", variant="stop")
                status_box = gr.Textbox(label="Status", value="⏸  Idle", interactive=False)

            # --- Right: live outputs ---
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Fitness Curve")
                chart = gr.Plot()

                gr.Markdown("### 🏆 Population Leaderboard")
                pop_table = gr.Dataframe(
                    headers=["Rank", "Fitness", "Origin", "Prompt Preview"],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )

        gr.Markdown("### 🔬 Best Prompt — What Changed This Generation?")
        gr.Markdown("Green = added words &nbsp;&nbsp; Red = removed words &nbsp;&nbsp; Gray = unchanged")
        diff_view = gr.HighlightedText(
            label="Prompt Diff (vs previous generation)",
            color_map={"added": "green", "removed": "red"},
            show_legend=True,
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Best Prompt (Full Text)")
                best_prompt_box = gr.Textbox(
                    label="",
                    lines=6,
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("### 🪵 Log")
                log_box = gr.Textbox(label="", lines=6, interactive=False)

        # Timer — polls every 3 seconds while open
        timer = gr.Timer(value=3)
        timer.tick(
            fn=poll_ui,
            outputs=[chart, pop_table, diff_view, best_prompt_box, log_box, status_box],
        )

        task.change(
            fn=lambda t: get_seed_prompt(t),
            inputs=[task],
            outputs=[seed_override],
        )

        btn_start.click(
            fn=start_run,
            inputs=[task, model, generations, pop_size, n_samples,
                    mutation_rate, elite_count, seed_override, use_judge],
            outputs=[status_box],
        )
        btn_stop.click(fn=stop_run, outputs=[status_box])

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
    )
