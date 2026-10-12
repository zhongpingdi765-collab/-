from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from jpsearch.bm25 import B, K1, bm25_term_score, idf
from jpsearch.corpus import Document, load_documents
from jpsearch.index import InvertedIndex, build_index
from jpsearch.snippet import make_snippet
from jpsearch.tokenizer import JapaneseTokenizer


@dataclass(frozen=True)
class Contribution:
    term: str
    tf: int
    df: int
    idf: float
    score: float


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    title: str
    body: str
    score: float
    snippet_text: str
    snippet_html: str
    contributions: tuple[Contribution, ...]


@dataclass(frozen=True)
class SearchResponse:
    query: str
    terms: tuple[str, ...]
    dropped: tuple[str, ...]
    results: tuple[SearchResult, ...]
    elapsed_ms: float

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "terms": list(self.terms),
            "dropped": list(self.dropped),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "results": [
                {
                    "id": result.doc_id,
                    "title": result.title,
                    "body": result.body,
                    "score": result.score,
                    "snippet_html": result.snippet_html,
                    "contributions": [
                        {
                            "term": item.term,
                            "tf": item.tf,
                            "df": item.df,
                            "idf": item.idf,
                            "score": item.score,
                        }
                        for item in result.contributions
                    ],
                }
                for result in self.results
            ],
        }


class SearchEngine:
    def __init__(self, documents: list[Document], tokenizer=None) -> None:
        self.documents = {document.id: document for document in documents}
        self.doc_order = [document.id for document in documents]
        self.tokenizer = tokenizer if tokenizer is not None else JapaneseTokenizer()
        self.index: InvertedIndex = build_index(documents, self.tokenizer)

    @classmethod
    def load(cls, path: Path | None = None, tokenizer=None) -> SearchEngine:
        return cls(load_documents(path), tokenizer=tokenizer)

    def search(self, query: str, k: int = 8) -> SearchResponse:
        started = time.perf_counter()
        limit = max(1, k)
        analyzed = self.tokenizer.analyze(query)
        terms = _unique(token.lemma for token in analyzed if token.keep)
        dropped = _unique(
            token.surface for token in analyzed if not token.keep and token.surface.strip()
        )
        results = self._rank(terms, limit) if terms else []
        elapsed_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(query, tuple(terms), tuple(dropped), tuple(results), elapsed_ms)

    def postings(self, word: str) -> tuple[str, tuple[tuple[Document, int], ...]] | None:
        lemmas = _unique(token.lemma for token in self.tokenizer.analyze(word) if token.keep)
        if len(lemmas) != 1 or lemmas[0] not in self.index.postings:
            return None
        lemma = lemmas[0]
        rows = []
        for posting in self.index.postings[lemma]:
            rows.append((self.documents[posting.doc_id], posting.tf))
        return lemma, tuple(rows)

    def stats(self) -> dict:
        return {
            "n_docs": self.index.n_docs,
            "vocabulary": self.index.vocabulary_size,
            "avgdl": round(self.index.avgdl, 2),
            "k1": K1,
            "b": B,
            "documents": [
                {"id": doc_id, "title": self.documents[doc_id].title} for doc_id in self.doc_order
            ],
        }

    def _rank(self, terms: list[str], limit: int) -> list[SearchResult]:
        scores: dict[str, float] = defaultdict(float)
        details: dict[str, list[Contribution]] = defaultdict(list)
        n_docs = self.index.n_docs

        for term in terms:
            postings = self.index.postings.get(term)
            if not postings:
                continue
            df = self.index.df[term]
            term_idf = idf(n_docs, df)
            for posting in postings:
                partial = bm25_term_score(
                    posting.tf,
                    self.index.doc_len[posting.doc_id],
                    self.index.avgdl,
                    term_idf,
                )
                scores[posting.doc_id] += partial
                details[posting.doc_id].append(
                    Contribution(term, posting.tf, df, term_idf, partial)
                )

        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.documents[item[0]].title),
        )
        query_lemmas = set(terms)
        results: list[SearchResult] = []
        for doc_id, score in ordered[:limit]:
            document = self.documents[doc_id]
            snippet = make_snippet(document.body, self.index.occurrences[doc_id], query_lemmas)
            contributions = tuple(sorted(details[doc_id], key=lambda item: -item.score))
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    title=document.title,
                    body=document.body,
                    score=score,
                    snippet_text=snippet.text,
                    snippet_html=snippet.html,
                    contributions=contributions,
                )
            )
        return results


def _unique(items) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered

