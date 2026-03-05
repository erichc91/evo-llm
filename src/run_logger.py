# run_logger.py — SQLite-backed run history for evo-llm experiments
# Each run gets its own .db file in runs/{run_id}.db
# Tracks: run config, generation-by-generation population + scores, best-per-gen history

import sqlite3
import json
import os
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_db_path(run_id: str) -> str:
    """Return absolute path to runs/{run_id}.db relative to project root."""
    return os.path.join(_PROJECT_ROOT, "runs", f"{run_id}.db")


def _connect(run_id: str) -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(run_id))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_run(config: dict) -> str:
    """
    IN:  config dict (task, model, pop_size, generations, mutation_rate, seed_prompt, …)
    OUT: run_id string
    """
    run_id = str(uuid.uuid4())
    runs_dir = os.path.join(_PROJECT_ROOT, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    conn = _connect(run_id)
    try:
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS run_config (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS generations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                generation   INTEGER NOT NULL,
                timestamp    TEXT    NOT NULL,
                best_fitness REAL    NOT NULL,
                mean_fitness REAL    NOT NULL,
                best_prompt  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS organisms (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                rank       INTEGER NOT NULL,
                prompt     TEXT    NOT NULL,
                fitness    REAL    NOT NULL,
                task       TEXT    NOT NULL
            );
        """)

        created_at = datetime.now(timezone.utc).isoformat()
        rows = [(k, json.dumps(v) if not isinstance(v, str) else v)
                for k, v in config.items()]
        rows.append(("created_at", created_at))
        cur.executemany(
            "INSERT OR REPLACE INTO run_config (key, value) VALUES (?, ?)", rows
        )
        conn.commit()
    finally:
        conn.close()

    return run_id


def log_generation(run_id: str, generation: int, scored_pop: list) -> None:
    """
    IN:  run_id, generation number, list of {prompt, fitness, task}
    OUT: None
    """
    if not scored_pop:
        scored_pop = [{"prompt": "", "fitness": 0.0, "task": ""}]

    sorted_pop = sorted(scored_pop, key=lambda x: x.get("fitness", 0.0), reverse=True)
    fitnesses = [o.get("fitness", 0.0) for o in sorted_pop]
    best_fitness = fitnesses[0]
    mean_fitness = sum(fitnesses) / len(fitnesses)
    best_prompt = sorted_pop[0].get("prompt", "")
    timestamp = datetime.now(timezone.utc).isoformat()

    conn = _connect(run_id)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO generations (generation, timestamp, best_fitness, mean_fitness, best_prompt) "
            "VALUES (?, ?, ?, ?, ?)",
            (generation, timestamp, best_fitness, mean_fitness, best_prompt),
        )
        cur.executemany(
            "INSERT INTO organisms (generation, rank, prompt, fitness, task) VALUES (?, ?, ?, ?, ?)",
            [
                (generation, rank + 1, o.get("prompt", ""), o.get("fitness", 0.0), o.get("task", ""))
                for rank, o in enumerate(sorted_pop)
            ],
        )
        conn.commit()
    finally:
        conn.close()


def get_best(run_id: str) -> dict:
    """
    IN:  run_id
    OUT: {prompt, fitness, generation}
    """
    conn = _connect(run_id)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT prompt, fitness, generation FROM organisms ORDER BY fitness DESC LIMIT 1"
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return {"prompt": "", "fitness": 0.0, "generation": 0}
    return {"prompt": row["prompt"], "fitness": row["fitness"], "generation": row["generation"]}


def get_fitness_curve(run_id: str) -> list:
    """
    IN:  run_id
    OUT: list of {generation, best_fitness, mean_fitness}
    """
    conn = _connect(run_id)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT generation, best_fitness, mean_fitness FROM generations ORDER BY generation"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {"generation": r["generation"], "best_fitness": r["best_fitness"], "mean_fitness": r["mean_fitness"]}
        for r in rows
    ]


def get_run_summary(run_id: str) -> dict:
    """
    IN:  run_id
    OUT: summary dict (config + fitness_curve + best)
    """
    conn = _connect(run_id)
    try:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM run_config")
        config = {r["key"]: r["value"] for r in cur.fetchall()}
    finally:
        conn.close()

    return {
        "run_id": run_id,
        "config": config,
        "fitness_curve": get_fitness_curve(run_id),
        "best": get_best(run_id),
    }


def list_runs(runs_dir: str = "runs") -> list:
    """
    IN:  runs directory path (default "runs", resolved relative to project root)
    OUT: list of {run_id, task, model, best_fitness, generations_completed, created_at}
    """
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.join(_PROJECT_ROOT, runs_dir)

    if not os.path.isdir(runs_dir):
        return []

    results = []
    for fname in os.listdir(runs_dir):
        if not fname.endswith(".db"):
            continue
        run_id = fname[:-3]
        db_path = os.path.join(runs_dir, fname)
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("SELECT key, value FROM run_config")
            cfg = {r["key"]: r["value"] for r in cur.fetchall()}

            cur.execute("SELECT COUNT(*) AS cnt FROM generations")
            gen_count = cur.fetchone()["cnt"]

            cur.execute("SELECT MAX(best_fitness) AS bf FROM generations")
            best_row = cur.fetchone()
            best_fitness = best_row["bf"] if best_row and best_row["bf"] is not None else 0.0

            conn.close()

            results.append({
                "run_id": run_id,
                "task": cfg.get("task", ""),
                "model": cfg.get("model", ""),
                "best_fitness": best_fitness,
                "generations_completed": gen_count,
                "created_at": cfg.get("created_at", ""),
            })
        except Exception:
            continue

    return results
