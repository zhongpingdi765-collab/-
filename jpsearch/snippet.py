from __future__ import annotations

import html
from dataclasses import dataclass

from jpsearch.tokenizer import Token

SENTENCE_END = "。！？"
MAX_SNIPPET_CHARS = 140


@dataclass(frozen=True)
class Snippet:
    text: str
    spans: tuple[tuple[int, int], ...]

    @property
    def html(self) -> str:
        return render_highlight(self.text, self.spans)


def make_snippet(body: str, tokens: list[Token], query_lemmas: set[str]) -> Snippet:
    if not body:
        return Snippet("", ())
    if not query_lemmas or not any(token.lemma in query_lemmas for token in tokens):
        return Snippet(_clip(body, 0, min(len(body), MAX_SNIPPET_CHARS), body), ())

    start, end = _best_span(body, tokens, query_lemmas)
    hits = [
        token
        for token in tokens
        if token.lemma in query_lemmas and token.start >= start and token.end <= end
    ]
    if hits and end - start > MAX_SNIPPET_CHARS:
        anchor = hits[0]
        half = MAX_SNIPPET_CHARS // 2
        start = max(start, anchor.start - half)
        end = min(end, start + MAX_SNIPPET_CHARS)
        if end - start < MAX_SNIPPET_CHARS:
            start = max(0, end - MAX_SNIPPET_CHARS)
    return Snippet(_clip(body, start, end, body), _relative_spans(body, tokens, query_lemmas, start, end))


def render_highlight(text: str, spans: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        start = max(start, cursor)
        end = min(max(end, start), len(text))
        if start >= end:
            continue
        parts.append(html.escape(text[cursor:start]))
        parts.append("<mark>" + html.escape(text[start:end]) + "</mark>")
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def _best_span(body: str, tokens: list[Token], query_lemmas: set[str]) -> tuple[int, int]:
    sentences = _sentence_spans(body)
    if not sentences:
        return 0, min(len(body), MAX_SNIPPET_CHARS)

    best_key: tuple[int, int, int] | None = None
    best_span = sentences[0]
    for index, (start, end) in enumerate(sentences):
        candidates = [(start, end)]
        if index + 1 < len(sentences):
            next_end = sentences[index + 1][1]
            if next_end - start <= MAX_SNIPPET_CHARS + 40:
                candidates.append((start, next_end))
        for cand_start, cand_end in candidates:
            coverage = len(_lemmas_inside(tokens, query_lemmas, cand_start, cand_end))
            key = (coverage, -(cand_end - cand_start), -cand_start)
            if best_key is None or key > best_key:
                best_key = key
                best_span = (cand_start, cand_end)
    return best_span


def _sentence_spans(body: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(body):
        if char in SENTENCE_END:
            if body[start : index + 1].strip():
                spans.append((start, index + 1))
            start = index + 1
    if start < len(body) and body[start:].strip():
        spans.append((start, len(body)))
    return spans


def _lemmas_inside(tokens: list[Token], query_lemmas: set[str], start: int, end: int) -> set[str]:
    return {
        token.lemma
        for token in tokens
        if token.lemma in query_lemmas and token.start >= start and token.end <= end
    }


def _relative_spans(
    body: str,
    tokens: list[Token],
    query_lemmas: set[str],
    start: int,
    end: int,
) -> tuple[tuple[int, int], ...]:
    prefix = 1 if start > 0 else 0
    spans: list[tuple[int, int]] = []
    for token in tokens:
        if token.lemma not in query_lemmas:
            continue
        if token.start < start or token.end > end:
            continue
        token_end = token.end
        if (
            token.pos.startswith("動詞")
            and token_end < end
            and token_end < len(body)
            and body[token_end] in "ただてで"
        ):
            token_end += 1
        spans.append((token.start - start + prefix, token_end - start + prefix))
    return tuple(_merge_spans(spans))


def _clip(body: str, start: int, end: int, original: str) -> str:
    text = body[start:end]
    if start > 0:
        text = "…" + text
    if end < len(original):
        text = text + "…"
    return text


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]
