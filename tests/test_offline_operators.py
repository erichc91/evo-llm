"""Tests for the dry-run offline mutation/crossover operators.

The property under test is the one `_offline_mutate`'s docstring claims and
that an earlier version silently violated: it must return a string DIFFERENT
from its input, for every mutation type, at every input length. If it can
return the input unchanged, dry-run exercises the plumbing but not the search,
and a broken selection step still looks fine.

Runnable two ways, deliberately:
    python tests/test_offline_operators.py     (direct)
    python -m pytest tests/                    (pytest)
A pytest-style file with no __main__ guard defines its tests, calls none of
them, and exits 0 — indistinguishable from success.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evolve_prompt import (  # noqa: E402
    _offline_crossover,
    _offline_mutate,
    _MUTATION_STYLES,
)

MUTATION_TYPES = list(_MUTATION_STYLES.keys()) + ["unknown_type"]

# 0 through 10 words, plus the awkward shapes that broke the first version.
LENGTH_CASES = [" ".join(f"w{i}" for i in range(n)) for n in range(11)]
SHAPE_CASES = [
    "",
    " ",
    "x",
    "ab",
    "one two",
    "You are careful.\n- Step one\n- Step two",
    "line one\n\nline three",
    "  leading and trailing  ",
]


def test_mutate_always_returns_a_different_string():
    random.seed(1234)
    for prompt in LENGTH_CASES + SHAPE_CASES:
        for mt in MUTATION_TYPES:
            out = _offline_mutate(prompt, mt)
            assert out != prompt, (
                f"_offline_mutate returned its input unchanged "
                f"(type={mt!r}, {len(prompt.split())} words, prompt={prompt!r})"
            )


def test_mutate_preserves_line_count_except_when_extending():
    """A bulleted prompt must not be flattened onto one line."""
    random.seed(7)
    prompt = "You are careful.\n- Step one\n- Step two"
    for mt in ("rephrase", "unknown_type"):
        out = _offline_mutate(prompt, mt)
        assert out.count("\n") == prompt.count("\n"), (
            f"{mt} collapsed newlines: {out!r}"
        )


def test_extend_varies_so_repeats_do_not_stack_one_sentence():
    random.seed(99)
    seen = {_offline_mutate("base prompt", "extend") for _ in range(40)}
    assert len(seen) > 1, "extend always appends the identical sentence"


def test_unknown_type_matches_rephrase_behaviour():
    """The real path falls back to rephrase for an unknown type; so must this."""
    prompt = "alpha beta gamma delta epsilon"
    random.seed(3)
    unknown = _offline_mutate(prompt, "unknown_type")
    random.seed(3)
    rephrase = _offline_mutate(prompt, "rephrase")
    assert unknown == rephrase


def test_crossover_keeps_content_from_both_parents():
    random.seed(5)
    out = _offline_crossover("one two", "three four")
    assert "one" in out and "four" in out, f"dropped a parent's content: {out!r}"
    for a, b in [("a", "b"), ("", "solo"), ("solo", ""), ("x y z", "p q r")]:
        merged = _offline_crossover(a, b)
        assert isinstance(merged, str) and merged.strip(), f"empty for {a!r}/{b!r}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if not tests:
        print("ERROR: collected zero tests")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
