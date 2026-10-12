from __future__ import annotations

import unittest

from jpsearch.corpus import Document
from jpsearch.engine import SearchEngine
from jpsearch.tokenizer import JapaneseTokenizer, Token


def _janome_available() -> bool:
    try:
        import janome
    except ImportError:
        return False
    return janome is not None


class WhitespaceTokenizer:
    def analyze(self, text: str) -> list[Token]:
        tokens: list[Token] = []
        cursor = 0
        for surface in text.split():
            start = text.find(surface, cursor)
            end = start + len(surface)
            cursor = end
            tokens.append(Token(surface, surface, "名詞,一般", start, end, True))
        return tokens

    def content_tokens(self, text: str) -> list[Token]:
        return self.analyze(text)


class RankingTest(unittest.TestCase):
    def test_query_with_both_terms_outranks_partial_match(self) -> None:
        documents = [
            Document("both", "両方", "台風 被害 台風"),
            Document("one", "片方", "台風 宇宙 観測"),
            Document("other", "無関係", "人工知能 研究"),
        ]
        engine = SearchEngine(documents, tokenizer=WhitespaceTokenizer())
        results = engine.search("台風 被害").results
        self.assertEqual(results[0].doc_id, "both")
        self.assertIn("one", [result.doc_id for result in results])
        self.assertNotIn("other", [result.doc_id for result in results])

    def test_title_boost_beats_a_single_body_mention(self) -> None:
        documents = [
            Document("title", "猫", "今日"),
            Document("body", "今日", "猫"),
        ]
        engine = SearchEngine(documents, tokenizer=WhitespaceTokenizer())
        results = engine.search("猫").results
        self.assertEqual(results[0].doc_id, "title")

    def test_contributions_sum_to_score(self) -> None:
        documents = [Document("d", "台風", "台風 被害")]
        engine = SearchEngine(documents, tokenizer=WhitespaceTokenizer())
        result = engine.search("台風 被害").results[0]
        self.assertAlmostEqual(result.score, sum(item.score for item in result.contributions))


@unittest.skipUnless(_janome_available(), "janome が必要です")
class JapaneseSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = JapaneseTokenizer()
        cls.engine = SearchEngine.load(tokenizer=cls.tokenizer)

    def test_inflection_returns_to_dictionary_form(self) -> None:
        lemmas = [token.lemma for token in self.tokenizer.content_tokens("選手が走った")]
        self.assertIn("走る", lemmas)

    def test_surfaces_cover_the_prepared_text(self) -> None:
        text = "台風の被害が広がった。"
        tokens = self.tokenizer.analyze(text)
        self.assertEqual("".join(token.surface for token in tokens), text)

    def test_particles_are_dropped(self) -> None:
        response = self.engine.search("の は が")
        self.assertEqual(response.terms, ())
        self.assertEqual(response.results, ())

    def test_typhoon_query_ranks_the_typhoon_article_first(self) -> None:
        results = self.engine.search("台風の被害").results
        self.assertEqual(results[0].doc_id, "typhoon")
        self.assertIn("snow", [result.doc_id for result in results])
        self.assertIn("台風", results[0].snippet_text)

    def test_expected_top_documents(self) -> None:
        expected = {
            "半導体工場": "semiconductor",
            "走った": "track",
            "浮世絵": "ukiyoe",
            "熱中症": "heatstroke",
            "古典籍": "classics",
            "海氷が減った": "arctic",
            "発酵食品": "fermentation",
            "レファレンス": "reference",
        }
        for query, doc_id in expected.items():
            with self.subTest(query=query):
                self.assertEqual(self.engine.search(query).results[0].doc_id, doc_id)

    def test_postings_list_typhoon(self) -> None:
        found = self.engine.postings("台風")
        self.assertIsNotNone(found)
        lemma, rows = found
        self.assertEqual(lemma, "台風")
        self.assertEqual([document.id for document, _tf in rows], ["typhoon"])


if __name__ == "__main__":
    unittest.main()
