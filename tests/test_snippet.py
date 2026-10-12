from __future__ import annotations

import unittest

from jpsearch.snippet import make_snippet, render_highlight
from jpsearch.tokenizer import Token


def token(surface: str, lemma: str, text: str, start: int | None = None) -> Token:
    if start is None:
        start = text.find(surface)
    return Token(surface, lemma, "名詞,一般", start, start + len(surface), True)


class SnippetTest(unittest.TestCase):
    def test_prefers_sentence_that_contains_query_terms(self) -> None:
        body = "無関係な文です。台風の被害が広がった。別の文です。"
        tokens = [
            token("台風", "台風", body),
            token("被害", "被害", body),
        ]
        snippet = make_snippet(body, tokens, {"台風", "被害"})
        self.assertIn("台風の被害", snippet.text)
        self.assertNotIn("無関係", snippet.text)
        self.assertIn("<mark>台風</mark>", snippet.html)
        self.assertIn("<mark>被害</mark>", snippet.html)

    def test_escapes_markup_around_highlights(self) -> None:
        text = '<script>台風</script>'
        start = text.find("台風")
        html = render_highlight(text, [(start, start + 2)])
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("<mark>台風</mark>", html)

    def test_noun_does_not_absorb_the_following_particle(self) -> None:
        body = "過去の台風でも同じ地域で川があふれた。"
        start = body.find("台風")
        tokens = [Token("台風", "台風", "名詞,一般", start, start + 2, True)]
        snippet = make_snippet(body, tokens, {"台風"})
        self.assertIn("<mark>台風</mark>でも", snippet.html)

    def test_highlight_includes_the_inflection_ending(self) -> None:
        body = "最後の直線を全力で走った。"
        start = body.find("走っ")
        tokens = [Token("走っ", "走る", "動詞,自立", start, start + 2, True)]
        snippet = make_snippet(body, tokens, {"走る"})
        self.assertIn("<mark>走った</mark>", snippet.html)

    def test_no_query_returns_opening(self) -> None:
        body = "冒頭だけを見せる文章です。後ろの文は切れます。"
        snippet = make_snippet(body, [], set())
        self.assertTrue(snippet.text.startswith("冒頭"))
        self.assertEqual(snippet.html.find("<mark>"), -1)


if __name__ == "__main__":
    unittest.main()
