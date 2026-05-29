from __future__ import annotations

from .models import CallRecord, Chunk, TranscriptTurn


def build_chunks(calls: list[CallRecord], max_chars: int = 900) -> list[Chunk]:
    chunks: list[Chunk] = []
    for call in calls:
        turns = call.turns or text_to_turns(call)
        current: list[TranscriptTurn] = []
        current_chars = 0
        index = 1
        for turn in turns:
            turn_chars = len(turn.text)
            if current and current_chars + turn_chars > max_chars:
                chunks.append(chunk_from_turns(call.call_id, index, current))
                index += 1
                current = []
                current_chars = 0
            current.append(turn)
            current_chars += turn_chars
        if current:
            chunks.append(chunk_from_turns(call.call_id, index, current))
    return chunks


def text_to_turns(call: CallRecord) -> list[TranscriptTurn]:
    lines = [line.strip() for line in call.transcript.splitlines() if line.strip()]
    if not lines:
        lines = [call.transcript]
    return [TranscriptTurn(call.call_id, index, "unknown", line) for index, line in enumerate(lines, start=1)]


def chunk_from_turns(call_id: str, index: int, turns: list[TranscriptTurn]) -> Chunk:
    text = "\n".join(f"turn:{turn.turn_index} speaker:{turn.speaker}: {turn.text}" for turn in turns)
    return Chunk(
        chunk_id=f"{call_id}:chunk-{index:03d}",
        call_id=call_id,
        turn_start=turns[0].turn_index,
        turn_end=turns[-1].turn_index,
        speaker_set=sorted({turn.speaker for turn in turns}),
        text=text,
        snippet=snippet(text),
        char_count=len(text),
    )


def snippet(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."

