"""Tests for :mod:`bitfrost.attribute_mapper`.

Strategy
--------
Pure-function tests against ``map_attributes``. Real fixtures captured from
``opentelemetry-instrumentation-openai`` and ``opentelemetry-instrumentation-anthropic``
live under ``tests/fixtures/`` and back the end-to-end mapping tests.

Coverage groups
---------------
- Skip / no-op (non-LLM spans)
- Provider extraction + casing normalization
- Model extraction with fallback
- Token usage including cache_read / cache_creation / reasoning
- Duration math (ns → integer ms)
- Status / outcome mapping (UNSET / OK / ERROR)
- Indexed prompts and completions extraction
- Real fixture end-to-end mapping
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bitfrost.attribute_mapper import map_attributes

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_args(**overrides: Any) -> dict[str, Any]:
    """Build the default kwargs dict for :func:`map_attributes`."""

    defaults: dict[str, Any] = {
        "span_name": "openai.chat",
        "attributes": {},
        "start_time_ns": 1_000_000_000,
        "end_time_ns": 2_000_000_000,
        "status_code": "UNSET",
        "status_description": None,
        "events": None,
        "instrumentation_scope_name": "opentelemetry.instrumentation.openai.v1",
    }
    defaults.update(overrides)
    return defaults


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Skip / no-op (non-LLM spans)
# ---------------------------------------------------------------------------


def test_returns_none_for_span_with_no_gen_ai_attributes() -> None:
    result = map_attributes(
        **_base_args(attributes={"http.method": "GET", "http.url": "https://example.com"})
    )
    assert result is None


def test_returns_none_for_span_with_empty_attributes() -> None:
    result = map_attributes(**_base_args(attributes={}))
    assert result is None


def test_returns_none_for_http_span_with_only_http_attrs() -> None:
    result = map_attributes(
        **_base_args(
            span_name="GET /api/users",
            attributes={
                "http.method": "GET",
                "http.status_code": 200,
                "url.full": "https://api.example.com/users",
            },
        )
    )
    assert result is None


# ---------------------------------------------------------------------------
# Provider extraction + casing normalization
# ---------------------------------------------------------------------------


def test_provider_extracted_from_gen_ai_system_lowercase() -> None:
    result = map_attributes(
        **_base_args(attributes={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o-mini"})
    )
    assert result is not None
    assert result["metadata"]["provider"] == "openai"


def test_provider_extracted_from_gen_ai_system_capitalized_normalized_to_lower() -> None:
    """Anthropic emits ``gen_ai.system = 'Anthropic'`` (capitalized).

    Mapper must lowercase it so dashboard provider filters match across the
    OpenAI ``'openai'`` and Anthropic ``'anthropic'`` providers consistently.
    """
    result = map_attributes(
        **_base_args(
            span_name="anthropic.chat",
            attributes={"gen_ai.system": "Anthropic", "gen_ai.request.model": "claude-haiku-4-5"},
        )
    )
    assert result is not None
    assert result["metadata"]["provider"] == "anthropic"


def test_provider_falls_back_to_ai_model_provider_when_gen_ai_system_missing() -> None:
    result = map_attributes(
        **_base_args(
            attributes={"ai.model.provider": "OpenAI.Responses", "ai.model.id": "gpt-4o-mini"},
        )
    )
    assert result is not None
    # ``OpenAI.Responses`` should normalize to base provider ``openai``
    assert result["metadata"]["provider"] == "openai"


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------


def test_model_extracted_from_gen_ai_request_model() -> None:
    result = map_attributes(
        **_base_args(attributes={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o-mini"})
    )
    assert result is not None
    assert result["model"] == "gpt-4o-mini"


def test_model_falls_back_to_ai_model_id_when_gen_ai_missing() -> None:
    result = map_attributes(
        **_base_args(
            attributes={"ai.model.provider": "openai", "ai.model.id": "gpt-4o-mini"},
        )
    )
    assert result is not None
    assert result["model"] == "gpt-4o-mini"


def test_response_model_surfaced_in_metadata_when_present() -> None:
    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.response.model": "gpt-4o-mini-2024-07-18",
            }
        )
    )
    assert result is not None
    assert result["metadata"]["responseModel"] == "gpt-4o-mini-2024-07-18"


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_tokens_input_output_total_summed_correctly() -> None:
    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.usage.input_tokens": 12,
                "gen_ai.usage.output_tokens": 7,
            }
        )
    )
    assert result is not None
    assert result["metadata"]["tokens"] == {"input": 12, "output": 7, "total": 19}


def test_tokens_cache_read_included_only_when_positive() -> None:
    result_zero = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.usage.input_tokens": 12,
                "gen_ai.usage.output_tokens": 1,
                "gen_ai.usage.cache_read_input_tokens": 0,
            }
        )
    )
    assert result_zero is not None
    assert "cache_read" not in result_zero["metadata"]["tokens"]

    result_pos = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.usage.input_tokens": 12,
                "gen_ai.usage.output_tokens": 1,
                "gen_ai.usage.cache_read_input_tokens": 8,
            }
        )
    )
    assert result_pos is not None
    assert result_pos["metadata"]["tokens"]["cache_read"] == 8


def test_tokens_cache_creation_included_only_when_positive() -> None:
    """Anthropic-specific: ``cache_creation_input_tokens`` only when > 0."""
    result_zero = map_attributes(
        **_base_args(
            span_name="anthropic.chat",
            attributes={
                "gen_ai.system": "Anthropic",
                "gen_ai.request.model": "claude-haiku-4-5",
                "gen_ai.usage.input_tokens": 13,
                "gen_ai.usage.output_tokens": 5,
                "gen_ai.usage.cache_creation_input_tokens": 0,
            },
        )
    )
    assert result_zero is not None
    assert "cache_creation" not in result_zero["metadata"]["tokens"]

    result_pos = map_attributes(
        **_base_args(
            span_name="anthropic.chat",
            attributes={
                "gen_ai.system": "Anthropic",
                "gen_ai.request.model": "claude-haiku-4-5",
                "gen_ai.usage.input_tokens": 13,
                "gen_ai.usage.output_tokens": 5,
                "gen_ai.usage.cache_creation_input_tokens": 42,
            },
        )
    )
    assert result_pos is not None
    assert result_pos["metadata"]["tokens"]["cache_creation"] == 42


def test_tokens_reasoning_extracted_from_llm_extension() -> None:
    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "o3-mini",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
                "llm.usage.reasoning_tokens": 200,
            }
        )
    )
    assert result is not None
    assert result["metadata"]["tokens"]["reasoning"] == 200


def test_tokens_fallback_to_ai_usage_namespace() -> None:
    result = map_attributes(
        **_base_args(
            attributes={
                "ai.model.provider": "openai",
                "ai.model.id": "gpt-4o-mini",
                "ai.usage.promptTokens": 12,
                "ai.usage.completionTokens": 7,
                "ai.usage.cachedInputTokens": 5,
            }
        )
    )
    assert result is not None
    tokens = result["metadata"]["tokens"]
    assert tokens["input"] == 12
    assert tokens["output"] == 7
    assert tokens["total"] == 19
    assert tokens["cache_read"] == 5


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


def test_duration_ms_is_integer_rounded_from_ns() -> None:
    """1775.4 ms (real OpenAI fixture timing) → 1775 ms (rounded int)."""
    result = map_attributes(
        **_base_args(
            attributes={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o-mini"},
            start_time_ns=1_000_000_000,
            end_time_ns=2_775_400_000,  # 1775.4 ms span
        )
    )
    assert result is not None
    assert result["durationMs"] == 1775
    assert isinstance(result["durationMs"], int)


def test_duration_zero_when_end_equals_start() -> None:
    result = map_attributes(
        **_base_args(
            attributes={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o-mini"},
            start_time_ns=1_000_000_000,
            end_time_ns=1_000_000_000,
        )
    )
    assert result is not None
    assert result["durationMs"] == 0


def test_duration_always_returns_int_never_float() -> None:
    """Voight ingest Zod schema rejects floats — mapper must always return int."""
    result = map_attributes(
        **_base_args(
            attributes={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o-mini"},
            start_time_ns=0,
            end_time_ns=500_000,  # 0.5 ms
        )
    )
    assert result is not None
    assert isinstance(result["durationMs"], int)
    assert result["durationMs"] >= 0


# ---------------------------------------------------------------------------
# Status / outcome
# ---------------------------------------------------------------------------


def test_outcome_success_when_status_unset() -> None:
    """OpenAI instrumentation leaves status UNSET on success — treat as success."""
    result = map_attributes(
        **_base_args(
            attributes={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o-mini"},
            status_code="UNSET",
        )
    )
    assert result is not None
    assert result["outcome"] == "success"
    assert "errorMessage" not in result


def test_outcome_success_when_status_ok() -> None:
    """Anthropic instrumentation sets status OK explicitly on success."""
    result = map_attributes(
        **_base_args(
            span_name="anthropic.chat",
            attributes={"gen_ai.system": "Anthropic", "gen_ai.request.model": "claude-haiku-4-5"},
            status_code="OK",
        )
    )
    assert result is not None
    assert result["outcome"] == "success"
    assert "errorMessage" not in result


def test_outcome_failed_when_status_error_with_message() -> None:
    result = map_attributes(
        **_base_args(
            attributes={"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o-mini"},
            status_code="ERROR",
            status_description="rate_limit_exceeded",
        )
    )
    assert result is not None
    assert result["outcome"] == "failed"
    assert result["errorMessage"] == "rate_limit_exceeded"


# ---------------------------------------------------------------------------
# Prompts / completions (indexed)
# ---------------------------------------------------------------------------


def test_indexed_prompts_concatenated_to_input_messages() -> None:
    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.prompt.0.role": "system",
                "gen_ai.prompt.0.content": "You are helpful.",
                "gen_ai.prompt.1.role": "user",
                "gen_ai.prompt.1.content": "Hello",
            }
        )
    )
    assert result is not None
    messages = result["input"]["messages"]
    assert messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]


def test_indexed_completions_extracted_to_response_text() -> None:
    """First completion's content lands in ``metadata.responseText``."""
    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": "Hello world",
                "gen_ai.completion.0.finish_reason": "stop",
            }
        )
    )
    assert result is not None
    assert result["metadata"]["responseText"] == "Hello world"


def test_finish_reason_extracted_from_first_completion() -> None:
    result = map_attributes(
        **_base_args(
            span_name="anthropic.chat",
            attributes={
                "gen_ai.system": "Anthropic",
                "gen_ai.request.model": "claude-haiku-4-5",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": "answer",
                "gen_ai.completion.0.finish_reason": "end_turn",
            },
        )
    )
    assert result is not None
    assert result["metadata"]["finishReason"] == "end_turn"


# ---------------------------------------------------------------------------
# Real fixture end-to-end mapping
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openai_fixture_span() -> dict[str, Any]:
    fixture = _load_fixture("openai_instrumentation_span.json")
    return fixture["spans"][0]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def anthropic_fixture_span() -> dict[str, Any]:
    fixture = _load_fixture("anthropic_instrumentation_span.json")
    return fixture["spans"][0]  # type: ignore[no-any-return]


def test_maps_real_openai_fixture_correctly(openai_fixture_span: dict[str, Any]) -> None:
    span = openai_fixture_span
    status_code_str = span["status"]["status_code"].split(".")[-1]  # "StatusCode.UNSET" → "UNSET"

    result = map_attributes(
        span_name=span["name"],
        attributes=span["attributes"],
        start_time_ns=span["start_time_ns"],
        end_time_ns=span["end_time_ns"],
        status_code=status_code_str,
        status_description=span["status"]["description"],
        events=span.get("events") or None,
        instrumentation_scope_name=span["instrumentation_scope"]["name"],
    )

    assert result is not None
    assert result["model"] == "gpt-4o-mini"
    assert result["metadata"]["provider"] == "openai"
    assert result["metadata"]["responseModel"] == "gpt-4o-mini-2024-07-18"
    assert result["outcome"] == "success"
    assert result["durationMs"] > 0
    assert isinstance(result["durationMs"], int)
    assert result["metadata"]["tokens"]["input"] == 12
    assert result["metadata"]["tokens"]["output"] == 1
    assert result["metadata"]["tokens"]["total"] == 13
    assert result["input"]["messages"] == [{"role": "user", "content": "Reply with exactly: pong"}]
    assert result["metadata"]["responseText"] == "pong"
    assert result["metadata"]["finishReason"] == "stop"


def test_maps_real_anthropic_fixture_correctly(anthropic_fixture_span: dict[str, Any]) -> None:
    span = anthropic_fixture_span
    status_code_str = span["status"]["status_code"].split(".")[-1]

    result = map_attributes(
        span_name=span["name"],
        attributes=span["attributes"],
        start_time_ns=span["start_time_ns"],
        end_time_ns=span["end_time_ns"],
        status_code=status_code_str,
        status_description=span["status"]["description"],
        events=span.get("events") or None,
        instrumentation_scope_name=span["instrumentation_scope"]["name"],
    )

    assert result is not None
    assert result["model"] == "claude-haiku-4-5"
    assert result["metadata"]["provider"] == "anthropic"
    assert result["metadata"]["responseModel"] == "claude-haiku-4-5-20251001"
    assert result["outcome"] == "success"
    assert result["durationMs"] > 0
    assert isinstance(result["durationMs"], int)
    assert result["metadata"]["tokens"]["input"] == 13
    assert result["metadata"]["tokens"]["output"] == 5
    assert result["metadata"]["tokens"]["total"] == 18
    assert result["input"]["messages"] == [{"role": "user", "content": "Reply with exactly: pong"}]
    assert result["metadata"]["responseText"] == "pong"
    assert result["metadata"]["finishReason"] == "end_turn"


# ---------------------------------------------------------------------------
# OTel GenAI semconv v1.32+ — modern instrumentation libraries
#
# Q1 2026 the OTel GenAI working group renamed several attributes for
# consistency with broader OTel resource-naming style. Modern
# instrumentation packages (opentelemetry-instrumentation-openai >=
# 0.60, anthropic >= 0.60, smolagents, LiteLLM modern) emit the new
# names below. Bitfrost reads BOTH generations on every field so an
# upgrade or a mixed-library codebase keeps working end-to-end. These
# tests pin the v1.32+ path; the existing tests above cover v1.27.
# ---------------------------------------------------------------------------


def test_provider_extracted_from_v132_provider_name() -> None:
    """``gen_ai.provider.name`` is the v1.32+ canonical key."""

    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
            }
        )
    )
    assert result is not None
    assert result["metadata"]["provider"] == "openai"


def test_provider_v132_takes_priority_over_v127_when_both_present() -> None:
    """If both names appear, v1.32+ wins so upgrades are forward-compatible."""

    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.provider.name": "anthropic",  # v1.32+
                "gen_ai.system": "openai",  # v1.27 — should be ignored
                "gen_ai.request.model": "claude-haiku-4-5",
            }
        )
    )
    assert result is not None
    assert result["metadata"]["provider"] == "anthropic"


def test_provider_v127_used_as_fallback_when_v132_absent() -> None:
    """Older instrumentation libraries still map cleanly via ``gen_ai.system``."""

    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
            }
        )
    )
    assert result is not None
    assert result["metadata"]["provider"] == "openai"


def test_input_messages_parsed_from_v132_json_blob() -> None:
    """``gen_ai.input.messages`` is a JSON string of typed-parts messages."""

    blob = '[{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]'
    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.input.messages": blob,
            }
        )
    )
    assert result is not None
    assert result["input"]["messages"] == [{"role": "user", "content": "Hello"}]


def test_output_messages_parsed_from_v132_with_finish_reasons() -> None:
    """``gen_ai.output.messages`` + sibling ``gen_ai.response.finish_reasons``.

    The finish reasons array aligns by index with the parsed messages.
    """

    blob = '[{"role": "assistant", "parts": [{"type": "text", "content": "Hi back"}]}]'
    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.output.messages": blob,
                "gen_ai.response.finish_reasons": ["stop"],
            }
        )
    )
    assert result is not None
    assert result["metadata"]["responseText"] == "Hi back"
    assert result["metadata"]["finishReason"] == "stop"


def test_inline_finish_reason_on_output_message_preserved() -> None:
    """Anthropic 0.60+ stamps ``finish_reason`` inline on each output message.

    When the message dict already carries one we don't overwrite from the
    sibling array — the inline value is more specific.
    """

    blob = (
        '[{"role": "assistant", '
        '"parts": [{"type": "text", "content": "pong"}], '
        '"finish_reason": "end_turn"}]'
    )
    result = map_attributes(
        **_base_args(
            span_name="anthropic.chat",
            attributes={
                "gen_ai.provider.name": "anthropic",
                "gen_ai.request.model": "claude-haiku-4-5",
                "gen_ai.output.messages": blob,
                "gen_ai.response.finish_reasons": ["stop"],  # less specific
            },
        )
    )
    assert result is not None
    assert result["metadata"]["finishReason"] == "end_turn"


def test_input_messages_falls_back_to_v127_indexed_when_blob_missing() -> None:
    """No ``gen_ai.input.messages`` → walk the indexed ``gen_ai.prompt.N.*``."""

    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.prompt.0.role": "user",
                "gen_ai.prompt.0.content": "Indexed legacy hello",
            }
        )
    )
    assert result is not None
    assert result["input"]["messages"] == [{"role": "user", "content": "Indexed legacy hello"}]


def test_malformed_messages_json_blob_falls_back_to_indexed_walk() -> None:
    """A broken JSON blob doesn't break ingestion — degrade to indexed walk."""

    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.input.messages": "{this is not valid JSON",
                "gen_ai.prompt.0.role": "user",
                "gen_ai.prompt.0.content": "Indexed survives malformed blob",
            }
        )
    )
    assert result is not None
    assert result["input"]["messages"] == [
        {"role": "user", "content": "Indexed survives malformed blob"}
    ]


def test_cache_read_extracted_from_v132_dotted_key() -> None:
    """``gen_ai.usage.cache_read.input_tokens`` is the v1.32+ dotted form."""

    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.provider.name": "anthropic",
                "gen_ai.request.model": "claude-haiku-4-5",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 20,
                "gen_ai.usage.cache_read.input_tokens": 75,
                "gen_ai.usage.cache_creation.input_tokens": 10,
            }
        )
    )
    assert result is not None
    assert result["metadata"]["tokens"]["cache_read"] == 75
    assert result["metadata"]["tokens"]["cache_creation"] == 10


def test_cache_read_v127_underscore_still_works_as_fallback() -> None:
    """Older libraries that still emit the underscore form keep mapping cleanly."""

    result = map_attributes(
        **_base_args(
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 20,
                "gen_ai.usage.cache_read_input_tokens": 50,
            }
        )
    )
    assert result is not None
    assert result["metadata"]["tokens"]["cache_read"] == 50


# ---------------------------------------------------------------------------
# Real v1.32+ fixtures — captured from instrumentation 0.60.0
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openai_fixture_span_v132() -> dict[str, Any]:
    fixture = _load_fixture("openai_span_v132.json")
    return fixture[0]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def anthropic_fixture_span_v132() -> dict[str, Any]:
    fixture = _load_fixture("anthropic_span_v132.json")
    return fixture[0]  # type: ignore[no-any-return]


def test_maps_real_v132_openai_fixture_correctly(
    openai_fixture_span_v132: dict[str, Any],
) -> None:
    """End-to-end against a real ``opentelemetry-instrumentation-openai==0.60.0`` span.

    This is the bug that produced ``provider: ""`` in production on
    2026-05-26 — the v1.27 mapper saw ``gen_ai.provider.name`` instead
    of ``gen_ai.system`` and defaulted to the empty string. The v1.32+
    multi-version fix turns this regression test green.
    """

    span = openai_fixture_span_v132
    status_code_str = span["status_code"].split(".")[-1]
    result = map_attributes(
        span_name=span["name"],
        attributes=span["attributes"],
        start_time_ns=span["start_time_ns"],
        end_time_ns=span["end_time_ns"],
        status_code=status_code_str,
        status_description=None,
        events=None,
        instrumentation_scope_name=span["instrumentation_scope"],
    )
    assert result is not None
    assert result["metadata"]["provider"] == "openai"
    assert result["model"] == "gpt-4o-mini"
    assert result["outcome"] == "success"
    assert isinstance(result["durationMs"], int)
    assert result["metadata"]["tokens"]["output"] > 0
    # The input messages JSON blob should parse cleanly.
    assert result["input"]["messages"][0]["role"] == "user"
    assert "Reply with exactly: pong" in result["input"]["messages"][0]["content"]
    # The output messages JSON blob + finish_reasons sibling should
    # produce a response text + finish reason.
    assert result["metadata"]["responseText"]
    assert result["metadata"]["finishReason"]


def test_maps_real_v132_anthropic_fixture_correctly(
    anthropic_fixture_span_v132: dict[str, Any],
) -> None:
    """End-to-end against a real ``opentelemetry-instrumentation-anthropic==0.60.0`` span."""

    span = anthropic_fixture_span_v132
    status_code_str = span["status_code"].split(".")[-1]
    result = map_attributes(
        span_name=span["name"],
        attributes=span["attributes"],
        start_time_ns=span["start_time_ns"],
        end_time_ns=span["end_time_ns"],
        status_code=status_code_str,
        status_description=None,
        events=None,
        instrumentation_scope_name=span["instrumentation_scope"],
    )
    assert result is not None
    assert result["metadata"]["provider"] == "anthropic"
    assert result["model"] == "claude-haiku-4-5"
    assert result["outcome"] == "success"
    assert result["metadata"]["tokens"]["input"] == 13
    assert result["metadata"]["tokens"]["output"] == 5
    assert result["input"]["messages"][0]["role"] == "user"
    assert result["metadata"]["responseText"] == "pong"
