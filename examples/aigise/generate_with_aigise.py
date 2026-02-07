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

import logging
import os
from typing import Any

from aigise.rl_integration import create as aigise_create

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

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
