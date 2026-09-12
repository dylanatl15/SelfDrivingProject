"""Evaluation: rollouts, the adversarial scenario suite, reports and the live viewer.

Names resolve lazily. Importing the submodules here eagerly makes
`python -m selfdrive.eval.evaluate` load evaluate.py twice and warn about it.
"""

from importlib import import_module

_EXPORTS = {
    "ConstantPolicy": "evaluate",
    "EvalResult": "evaluate",
    "RandomPolicy": "evaluate",
    "run_episodes": "evaluate",
    "Scenario": "scenarios",
    "all_scenarios": "scenarios",
    "format_suite": "scenarios",
    "run_suite": "scenarios",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
