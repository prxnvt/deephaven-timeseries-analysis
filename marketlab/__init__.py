"""marketlab — backtest engine and (soon) generative market-model tooling.

Pure Python: pandas/numpy (and, in later milestones, torch). This package must
never import `deephaven.*` — it is imported both by the Deephaven dashboard
scripts inside the container and by host-side training/eval/test tooling.
"""

__version__ = "0.1.0"
