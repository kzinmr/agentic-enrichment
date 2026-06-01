from __future__ import annotations

from typing import Any


# Keys whose values may carry raw transcript text. Replay artifacts strip these by
# default so a shared/replayable trace never leaks raw call content (the core RLM
# tool-boundary invariant: raw transcripts never travel outside fetch_chunks/run_sql).
SENSITIVE_TEXT_KEYS = frozenset(
    {"snippet", "text", "best_text", "transcript", "calllog_result", "transcription_summary"}
)

REDACTED = "[redacted]"


def redact_for_replay(value: Any, *, sensitive_keys: frozenset[str] = SENSITIVE_TEXT_KEYS) -> Any:
    """Return a deep copy of ``value`` with raw-text fields replaced by a marker.

    Structural fields (ids, scores, counts, terms, latency, tokens) are preserved so the
    artifact stays useful for replay/debugging; only free-text transcript content is removed.
    """
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if key in sensitive_keys and item not in (None, "", [], {}):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_for_replay(item, sensitive_keys=sensitive_keys)
        return redacted
    if isinstance(value, list):
        return [redact_for_replay(item, sensitive_keys=sensitive_keys) for item in value]
    return value
