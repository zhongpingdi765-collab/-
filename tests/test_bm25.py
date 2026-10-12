from __future__ import annotations

import math
import unittest

from jpsearch.bm25 import bm25_term_score, idf


class Bm25Test(unittest.TestCase):
    def test_idf_matches_formula(self) -> None:
        n_docs, df = 4, 1
        expected = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self.assertAlmostEqual(idf(n_docs, df), expected)

    def test_rarer_term_has_higher_idf(self) -> None:
        self.assertGreater(idf(10, 1), idf(10, 8))

    def test_longer_document_scores_lower_for_the_same_count(self) -> None:
        term_idf = idf(10, 2)
        short = bm25_term_score(tf=2, doc_len=20, avgdl=40, term_idf=term_idf)
        long = bm25_term_score(tf=2, doc_len=80, avgdl=40, term_idf=term_idf)
        self.assertGreater(short, long)

    def test_repeated_term_increases_score_with_diminishing_effect(self) -> None:
        term_idf = idf(10, 2)
        once = bm25_term_score(1, 40, 40, term_idf)
        twice = bm25_term_score(2, 40, 40, term_idf)
        thrice = bm25_term_score(3, 40, 40, term_idf)
        self.assertGreater(twice, once)
        self.assertGreater(twice - once, thrice - twice)

    def test_zero_count_is_zero(self) -> None:
        self.assertEqual(bm25_term_score(0, 10, 10, idf(5, 1)), 0.0)


if __name__ == "__main__":
    unittest.main()
