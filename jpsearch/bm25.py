from __future__ import annotations

import math

K1 = 1.5
B = 0.75
TITLE_BOOST = 2


def idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_term_score(
    tf: int,
    doc_len: int,
    avgdl: float,
    term_idf: float,
    k1: float = K1,
    b: float = B,
) -> float:
    if tf <= 0 or term_idf == 0.0:
        return 0.0
    average = avgdl if avgdl > 0 else 1.0
    length_norm = 1.0 - b + b * (doc_len / average)
    return term_idf * (tf * (k1 + 1.0)) / (tf + k1 * length_norm)
