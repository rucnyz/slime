"""
AIgiSE Integration for slime Training

This module provides the main ``generate`` function that slime calls during
rollout.  It mirrors the structure of ``examples/tau-bench/generate_with_tau.py``
but replaces the tau-bench agent/environment with AIgiSE's Evaluation system.

The key idea:
    1. slime calls ``generate(args, sample, sampling_params)``
    2. We create an ``aigise.rl_integration.Client`` (once, cached)
    3. For each sample we open an ``RLSession``, which creates a ``SlimeLlm``
       and injects it into the AIgiSE agent
    4. The agent runs in its sandbox environment, with every LLM call routed
       to slime's sglang server and token-tracked
    5. The resulting tokens/loss_mask/reward are written back into the
       slime ``Sample``

Configuration:
    Edit AIGISE_CONFIGS below or override via environment variables.
"""

import asyncio
import logging
import os
from typing import Any

# ---------------------------------------------------------------------------
# Debug logging setup — MUST run before importing aigise (which calls
# setup_aigise_logging() on import) and before slime's configure_logger()
# which uses force=True and would override our settings.
#
# Set AIGISE_LOG_LEVEL=DEBUG in the shell or RUNTIME_ENV_JSON to enable.
# ---------------------------------------------------------------------------
_aigise_log_level = os.environ.get("AIGISE_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _aigise_log_level, logging.INFO)

logging.basicConfig(
    level=_log_level,
    format="[%(asctime)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# ---------------------------------------------------------------------------
# Persistent file logging — Worker logs go to /dev/pts (tmux terminal) which
# gets overwhelmed by Ray stats dumps.  Write AIgiSE logs to a file so they
# survive regardless of terminal buffer state.
# ---------------------------------------------------------------------------
_aigise_worker_log = os.environ.get("AIGISE_WORKER_LOG", "/root/aigise_worker.log")
_file_handler = logging.FileHandler(_aigise_worker_log, mode="a")
_file_handler.setLevel(_log_level)
_file_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(name)s:%(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logging.getLogger().addHandler(_file_handler)

from aigise.rl_integration import create as aigise_create

from slime.utils.types import Sample

logger = logging.getLogger(__name__)
logger.info(f"AIgiSE log level: {_aigise_log_level}")

# ---------------------------------------------------------------------------
# Configuration — edit these for your setup
# ---------------------------------------------------------------------------

AIGISE_CONFIGS = {
    # Name of the agent directory under aigise/examples/agents/
    "agent_name": os.environ.get("AIGISE_AGENT_NAME", "poc_agent_static_tools"),
    # Name of the benchmark (maps to an Evaluation subclass via registry)
    "benchmark_name": os.environ.get("AIGISE_BENCHMARK_NAME", "cybergym"),
}

# ---------------------------------------------------------------------------
# Cached client (created once per process, reused across samples)
# ---------------------------------------------------------------------------

_client = None

# ---------------------------------------------------------------------------
# Concurrency limiter — SeCodePLT builds a unique ~4-5GB Docker image per
# task.  When dozens of tasks launch simultaneously they all hit
# ensure_docker_image() and the Docker daemon serialises the builds, causing
# a deadlock-like stall.  Limiting concurrency keeps resource usage sane.
#
# Set AIGISE_MAX_CONCURRENT to control the limit (default 4).
# ---------------------------------------------------------------------------
_eval_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Return (and lazily create) the per-process concurrency semaphore."""
    global _eval_semaphore
    if _eval_semaphore is None:
        max_concurrent = int(os.environ.get("AIGISE_MAX_CONCURRENT", "4"))
        _eval_semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"AIgiSE concurrency limit: {max_concurrent}")
    return _eval_semaphore


def _get_client():
    global _client
    if _client is None:
        _client = aigise_create(
            agent_name=AIGISE_CONFIGS["agent_name"],
            benchmark_name=AIGISE_CONFIGS["benchmark_name"],
        )
        logger.info(f"Created AIgiSE client: agent={AIGISE_CONFIGS['agent_name']}, benchmark={AIGISE_CONFIGS['benchmark_name']}")
    return _client


# ---------------------------------------------------------------------------
# Main entry point for slime
# ---------------------------------------------------------------------------


async def generate(
    args: dict[str, Any],
    sample: Sample,
    sampling_params: dict,
) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for AIgiSE.

    This is the main entry point for slime training.  It is referenced via:
        --custom-generate-function-path generate_with_aigise.generate

    Args:
        args: Rollout arguments from slime training pipeline.
              Must have: sglang_router_ip, sglang_router_port, partial_rollout,
                         hf_checkpoint (for tokenizer)
        sample: Sample containing task data.
                - sample.prompt: the task index (int, as string) into the dataset
                - sample.metadata: the full dataset row (dict)
        sampling_params: LLM sampling parameters (temperature, max_new_tokens, etc.)

    Returns:
        Sample object containing the complete interaction trajectory with
        tokens, loss_mask, response, reward, and status.

    Raises:
        AssertionError: If partial_rollout is requested (not supported)
    """
    # AIgiSE does full-trajectory rollout, no partial rollout support
    assert not args.partial_rollout, "Partial rollout is not supported for AIgiSE interactions."

    task_index = sample.prompt

    async with _get_semaphore():
        logger.info(f"Starting AIgiSE agent interaction for task {task_index}")

        client = _get_client()

        with client.init_session() as session:
            result_sample = await session.slime_generate(
                args=args,
                sample=sample,
                sampling_params=sampling_params,
            )

        logger.info(f"Finished AIgiSE interaction for task {task_index}: status={result_sample.status}, tokens={len(result_sample.tokens) if result_sample.tokens else 0}, response_length={result_sample.response_length}")

    return result_sample
