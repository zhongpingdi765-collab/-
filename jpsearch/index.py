from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from jpsearch.bm25 import TITLE_BOOST
from jpsearch.corpus import Document
from jpsearch.tokenizer import Token


@dataclass(frozen=True)
class Posting:
    doc_id: str
    tf: int


@dataclass
class InvertedIndex:
    postings: dict[str, list[Posting]] = field(default_factory=dict)
    df: dict[str, int] = field(default_factory=dict)
    doc_len: dict[str, int] = field(default_factory=dict)
    occurrences: dict[str, list[Token]] = field(default_factory=dict)
    avgdl: float = 0.0
    n_docs: int = 0

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)


def build_index(documents: list[Document], tokenizer, title_boost: int = TITLE_BOOST) -> InvertedIndex:
    index = InvertedIndex(n_docs=len(documents))
    if not documents:
        return index

    buckets: dict[str, list[Posting]] = defaultdict(list)
    lengths: dict[str, int] = {}
    occurrences: dict[str, list[Token]] = {}

    for document in documents:
        body_tokens = tokenizer.content_tokens(document.body)
        title_tokens = tokenizer.content_tokens(document.title)
        tf: dict[str, int] = defaultdict(int)
        for token in body_tokens:
            tf[token.lemma] += 1
        for token in title_tokens:
            tf[token.lemma] += title_boost
        lengths[document.id] = len(body_tokens) + title_boost * len(title_tokens)
        occurrences[document.id] = body_tokens
        for lemma, count in tf.items():
            buckets[lemma].append(Posting(document.id, count))

    index.postings = {
        lemma: sorted(postings, key=lambda posting: posting.doc_id)
        for lemma, postings in buckets.items()
    }
    index.df = {lemma: len(postings) for lemma, postings in index.postings.items()}
    index.doc_len = lengths
    index.occurrences = occurrences
    index.avgdl = sum(lengths.values()) / len(documents)
    return index
