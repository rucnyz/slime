"""
AIgiSE Generate Function for slime rollout.

This module provides a generate function compatible with slime's rollout system,
allowing integration of AIgiSE's multi-turn agent capabilities.

Usage in slime rollout config:
    generate_func: examples.aigise.generate_with_aigise.generate

Environment variables:
    AIGISE_AGENT_NAME: Agent name defined in examples/agents/ (default: vul_agent_static_tools)
    AIGISE_BENCHMARK_NAME: Benchmark name defined in evaluations/ (default: secodeplt)
"""

import os
from typing import Any

import aigise
from slime.utils.types import Sample

_aigise_client = None


def _get_client(api_base: str, model_name: str):
    """Get or create the AIgiSE client singleton."""
    global _aigise_client
    if _aigise_client is None:
        benchmark_name = os.getenv("AIGISE_BENCHMARK_NAME", "secodeplt")
        agent_name = os.getenv("AIGISE_AGENT_NAME", "vul_agent_static_tools")
        _aigise_client = aigise.create(agent_name, benchmark_name, api_base, model_name)
    return _aigise_client


async def generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """Generate using AIgiSE agent with multi-turn conversation.

    This function integrates AIgiSE's agent framework into slime's rollout system.
    The agent and benchmark are configured via environment variables.

    Args:
        args: Rollout arguments containing:
            - sglang_router_ip: SGLang router IP
            - sglang_router_port: SGLang router port
            - model_name: Model name
        sample: Sample object with prompt and metadata
        sampling_params: Sampling parameters (temperature, max_new_tokens, etc.)

    Returns:
        Updated Sample object with response and status
    """
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    client = _get_client(url, args.model_name)
    with client.init_session() as session:
        sample = await session.slime_generate(args, sample, sampling_params)
    return sample

