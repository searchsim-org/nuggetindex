"""Provider-agnostic LLM client plumbing.

Individual provider clients (OpenAI, Anthropic, Google, Ollama, OpenRouter,
OpenAI-compat) live next to this file and are imported lazily by
``build_client`` so installing the package without a given provider extra
does not break imports of this subpackage.
"""

from nuggetindex.extractors.clients.base import LLMClient, LLMConfig, build_client

__all__ = ["LLMClient", "LLMConfig", "build_client"]
