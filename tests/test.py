"""Tiered automated tests for Project 2 — Panda **Push** from Pixels.

Grading rubric (cumulative — you earn a tier only if all lower tiers pass):

    5 pts — your notebook trains an allowed SB3 agent (imports one of A2C / PPO / DDPG / TD3 / SAC and
            calls ``.learn(...)`` in YOUR sections, not the prewritten naive baseline), AND the submitted
            ``model.pt`` loads, satisfies the I/O contract, has <= PARAM_LIMIT parameters, and is fast
            enough on CPU.
   10 pts — + the agent MOVES the cube:   touch_rate   >= TOUCH_RATE_THRESHOLD  (0.80).
   15 pts — + the agent SOLVES the push:  success_rate >= SUCCESS_RATE_THRESHOLD (0.50).

These call into the installed ``panda_push_pixels`` package, so the contract, environment, and grading
logic are identical to what the instructor runs. Only the (hidden) evaluation seed offset and the number
of episodes differ at final grading.

Run locally:   cd template && pytest tests/test.py -v
Point a run at a specific model / notebook:  MODEL_PATH=... NOTEBOOK_PATH=... pytest tests/test.py -v
"""

import hashlib
import os

import pytest

from panda_push_pixels import contract, grading

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pt")

# The working notebook: student's ``project.ipynb`` by default; fall back to the instructor's
# ``project_completed.ipynb`` (or whatever NOTEBOOK_PATH points at).
NOTEBOOK_PATH = next(
    (p for p in (os.environ.get("NOTEBOOK_PATH"), "project.ipynb", "project_completed.ipynb")
     if p and os.path.isfile(p)),
    "project.ipynb",
)


@pytest.fixture(scope="session")
def metrics(request):
    """Evaluate the submitted model ONCE (touch_rate + success_rate come from the same episodes).

    Cached on disk under .pytest_cache/ (already gitignored), keyed by model.pt's content hash.
    GitHub Classroom's autograding scores Tier 10 and Tier 15 as two separate ``pytest`` processes
    (so each gets its own partial-credit line) — without this cache, that would re-run the 30
    evaluation episodes twice. The second process finds a hit (same model.pt) and reuses it.
    """
    grading.check_contract(MODEL_PATH)   # loadable + I/O contract before we spend episodes on it
    with open(MODEL_PATH, "rb") as f:
        fingerprint = hashlib.sha256(f.read()).hexdigest()

    cache_key = "panda_push_pixels/eval_metrics"
    cached = request.config.cache.get(cache_key, None)
    if cached is not None and cached.get("fingerprint") == fingerprint:
        return cached["metrics"]

    result = grading.evaluate(MODEL_PATH, n_episodes=contract.EVAL_EPISODES_CI)
    request.config.cache.set(cache_key, {"fingerprint": fingerprint, "metrics": result})
    return result


# ===========================================================================
# Tier 5 — the notebook trains an SB3 agent, and model.pt meets the contract
# ===========================================================================
class TestTier05_TrainingAndContract:
    def test_notebook_trains_an_sb3_agent(self):
        ok, detail = grading.notebook_trains_sb3(NOTEBOOK_PATH)
        assert ok, (
            f"[{NOTEBOOK_PATH}] must import one of {contract.ALLOWED_SB3_ALGOS} and call .learn() in "
            f"YOUR sections (>= section 3) — the prewritten naive baseline does not count. {detail}"
        )

    def test_model_file_exists(self):
        assert os.path.isfile(MODEL_PATH), (
            f"{MODEL_PATH} not found — export it from your notebook (panda_push_pixels.export_model) "
            f"and commit it to your repo root."
        )

    def test_model_loads_as_torchscript(self):
        grading.load_policy(MODEL_PATH)   # must be CPU TorchScript (use panda_push_pixels.export_model)

    def test_io_contract(self):
        # float32 (B, 12, 112, 112) in [0,255] -> (B, 7); finite; batch-safe.
        grading.check_contract(MODEL_PATH)

    def test_parameter_limit(self):
        n = grading.count_parameters(MODEL_PATH)
        assert n <= contract.PARAM_LIMIT, f"model has {n:,} parameters > limit {contract.PARAM_LIMIT:,}"

    def test_inference_latency(self):
        dt = grading.measure_latency(MODEL_PATH)
        assert dt <= contract.LATENCY_BUDGET_S, (
            f"forward pass {dt * 1000:.1f} ms > budget {contract.LATENCY_BUDGET_S * 1000:.0f} ms"
        )


# ===========================================================================
# Tier 10 — the agent moves the cube
# ===========================================================================
class TestTier10_Touch:
    def test_touch_rate(self, metrics):
        assert metrics["touch_rate"] >= contract.TOUCH_RATE_THRESHOLD, (
            f"touch_rate {metrics['touch_rate']:.0%} < threshold {contract.TOUCH_RATE_THRESHOLD:.0%} "
            f"over {metrics['n_episodes']} episodes — the cube must move > "
            f"{contract.TOUCH_DISPLACEMENT * 100:.0f} cm from its spawn in that fraction of episodes."
        )


# ===========================================================================
# Tier 15 — the agent solves the push (cube reaches the target and dwells)
# ===========================================================================
class TestTier15_PushSuccess:
    def test_success_rate(self, metrics):
        assert metrics["success_rate"] >= contract.SUCCESS_RATE_THRESHOLD, (
            f"success_rate {metrics['success_rate']:.0%} < threshold "
            f"{contract.SUCCESS_RATE_THRESHOLD:.0%} over {metrics['n_episodes']} episodes "
            f"(touch_rate={metrics['touch_rate']:.0%}, median_reward={metrics['median_reward']:.1f})."
        )
