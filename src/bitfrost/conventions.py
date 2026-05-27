"""OpenTelemetry semantic convention constants for LLM spans.

Three namespaces are recognized:

- ``gen_ai.*`` — the OpenTelemetry GenAI semantic conventions (primary).
- ``ai.*`` — the Vercel AI SDK convention (fallback, mostly for cross-runtime
  parity with the TypeScript Vercel AI SDK exporter).
- ``llm.*`` — Traceloop's OpenLLMetry extension attributes that
  ``opentelemetry-instrumentation-openai`` and
  ``opentelemetry-instrumentation-anthropic`` add on top of the gen_ai
  semconv (total_tokens, reasoning_tokens, is_streaming, request type).

The mapper uses ``gen_ai.*`` as the primary path and falls back to ``ai.*``
on a per-field basis. ``llm.*`` extension attributes are read additively.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# OTel GenAI semantic conventions (primary)
# ---------------------------------------------------------------------------

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"

GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read_input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation_input_tokens"

# Indexed prompts and completions (v1.27 — pre-2026):
#   gen_ai.prompt.0.role / gen_ai.prompt.0.content
#   gen_ai.prompt.1.role / gen_ai.prompt.1.content
#   gen_ai.completion.0.role / gen_ai.completion.0.content / .finish_reason
GEN_AI_PROMPT_PREFIX = "gen_ai.prompt."
GEN_AI_COMPLETION_PREFIX = "gen_ai.completion."


# ---------------------------------------------------------------------------
# OTel GenAI semantic conventions v1.32+ (Q1 2026)
# ---------------------------------------------------------------------------
#
# The semconv shipped breaking renames between v1.27 and v1.32 to align
# with OTel's broader resource-naming style. Modern instrumentation
# libraries (opentelemetry-instrumentation-openai >= 0.60, anthropic
# >= 0.60, smolagents, LiteLLM >= 1.50) emit the v1.32 names below.
# Older libraries still emit the v1.27 names above.
#
# The mapper reads BOTH generations: v1.32 takes priority on a per-field
# basis, then falls back to v1.27 so older spans still work end-to-end.
# This is the same pattern Logfire / Phoenix / Langfuse / Braintrust use
# — Bitfrost is OTel-spec-native, not pinned to one instrumentation
# version.

# Renamed: gen_ai.system → gen_ai.provider.name
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"

# New: explicit operation name (e.g. "chat", "embeddings", "completion").
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

# New canonical streaming flag (was llm.is_streaming in v1.27).
GEN_AI_IS_STREAMING = "gen_ai.is_streaming"

# Renamed: indexed `gen_ai.prompt.N.*` and `gen_ai.completion.N.*`
# collapsed into single JSON-string attributes. Format per spec:
#   "[{\"role\": \"user\", \"parts\": [{\"type\": \"text\", \"content\": \"...\"}]}]"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"

# Renamed (extra dot): gen_ai.usage.cache_read_input_tokens
#                    → gen_ai.usage.cache_read.input_tokens
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS_DOTTED = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS_DOTTED = (
    "gen_ai.usage.cache_creation.input_tokens"
)


# ---------------------------------------------------------------------------
# Vercel AI SDK convention (fallback, ai.*)
# ---------------------------------------------------------------------------

AI_MODEL_ID = "ai.model.id"
AI_MODEL_PROVIDER = "ai.model.provider"

AI_USAGE_PROMPT_TOKENS = "ai.usage.promptTokens"
AI_USAGE_COMPLETION_TOKENS = "ai.usage.completionTokens"
AI_USAGE_CACHED_INPUT_TOKENS = "ai.usage.cachedInputTokens"
AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "ai.usage.cacheCreationInputTokens"

AI_PROMPT_MESSAGES = "ai.prompt.messages"
AI_RESPONSE_TEXT = "ai.response.text"
AI_RESPONSE_TOOL_CALLS = "ai.response.toolCalls"
AI_RESPONSE_FINISH_REASON = "ai.response.finishReason"

# Vercel AI SDK per-user telemetry metadata (Voight maps these to metadata.tags):
#   ai.telemetry.metadata.<key>
AI_TELEMETRY_METADATA_PREFIX = "ai.telemetry.metadata."


# ---------------------------------------------------------------------------
# Traceloop OpenLLMetry extensions (llm.*)
# ---------------------------------------------------------------------------

LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
LLM_USAGE_REASONING_TOKENS = "llm.usage.reasoning_tokens"

LLM_IS_STREAMING = "llm.is_streaming"
LLM_REQUEST_TYPE = "llm.request.type"
LLM_HEADERS = "llm.headers"


# ---------------------------------------------------------------------------
# Span name prefixes (per-library)
# ---------------------------------------------------------------------------

# opentelemetry-instrumentation-openai emits spans named "openai.chat"
# (and similar for "openai.embeddings", "openai.images", etc.)
SPAN_NAME_OPENAI_PREFIX = "openai."

# opentelemetry-instrumentation-anthropic emits "anthropic.chat".
SPAN_NAME_ANTHROPIC_PREFIX = "anthropic."

# Vercel AI SDK (TS) emits "ai.streamText", "ai.generateText.doGenerate", etc.
SPAN_NAME_VERCEL_PREFIX = "ai."


__all__ = [
    "AI_MODEL_ID",
    "AI_MODEL_PROVIDER",
    "AI_PROMPT_MESSAGES",
    "AI_RESPONSE_FINISH_REASON",
    "AI_RESPONSE_TEXT",
    "AI_RESPONSE_TOOL_CALLS",
    "AI_TELEMETRY_METADATA_PREFIX",
    "AI_USAGE_CACHED_INPUT_TOKENS",
    "AI_USAGE_CACHE_CREATION_INPUT_TOKENS",
    "AI_USAGE_COMPLETION_TOKENS",
    "AI_USAGE_PROMPT_TOKENS",
    "GEN_AI_COMPLETION_PREFIX",
    "GEN_AI_INPUT_MESSAGES",
    "GEN_AI_IS_STREAMING",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_OUTPUT_MESSAGES",
    "GEN_AI_PROMPT_PREFIX",
    "GEN_AI_PROVIDER_NAME",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_P",
    "GEN_AI_RESPONSE_FINISH_REASONS",
    "GEN_AI_RESPONSE_ID",
    "GEN_AI_RESPONSE_MODEL",
    "GEN_AI_SYSTEM",
    "GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS",
    "GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS_DOTTED",
    "GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS",
    "GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS_DOTTED",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "LLM_HEADERS",
    "LLM_IS_STREAMING",
    "LLM_REQUEST_TYPE",
    "LLM_USAGE_REASONING_TOKENS",
    "LLM_USAGE_TOTAL_TOKENS",
    "SPAN_NAME_ANTHROPIC_PREFIX",
    "SPAN_NAME_OPENAI_PREFIX",
    "SPAN_NAME_VERCEL_PREFIX",
]
