# Changelog

All notable changes to `bitfrost` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-08

First public release.

### Added

- **OpenTelemetry SpanExporter** (`BitfrostExporter`) that maps GenAI spans to a
  normalized event shape, supporting both the v1.27 and v1.32+ semantic-convention
  attribute generations.
- **Five backends**: `ConsoleBackend` (rich terminal), `SQLiteBackend` (persistent
  local log, WAL mode), `JSONLBackend` (replay-able), `OTLPBackend` (JSON over HTTP
  to any collector/webhook), and `VoightBackend` (optional hosted dashboards), plus
  `TeeBackend` to fan out to several at once.
- **Auto-instrument helpers**: `instrument_openai`, `instrument_anthropic`,
  `instrument_litellm`, `instrument_smolagents`, `instrument_auto`, and `quickstart`.
- **CLI**: `bitfrost watch / replay / query / vacuum / tui / serve`.
- **Interactive TUI** (`bitfrost tui`) and an **offline web dashboard**
  (`bitfrost serve`) with live charts, filters, per-span detail, and an SSE feed.
- **Privacy**: three levels (`minimal` / `standard` / `full`) with PII scrubbing
  (12 patterns + credit-card Luhn) applied in-process before export.
- **Embedded pricing** for local cost estimates (top models; Decimal arithmetic).

### Notes

- Python 3.10–3.13. MIT licensed.
- The core has no dependency on any hosted service; Voight is opt-in.
