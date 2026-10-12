from __future__ import annotations

from dataclasses import dataclass

from jpsearch.textutil import prepare_text

STOPWORDS = frozenset(
    {
        "する",
        "いる",
        "ある",
        "なる",
        "れる",
        "られる",
        "できる",
        "くる",
        "いく",
        "おく",
        "しまう",
        "くださる",
        "いただく",
        "こと",
        "もの",
        "ため",
        "よう",
        "さん",
        "これ",
        "それ",
        "あれ",
        "どれ",
        "ここ",
        "そこ",
        "あそこ",
        "どこ",
        "とき",
        "ところ",
        "うち",
        "ほど",
        "くらい",
        "など",
        "および",
        "として",
        "により",
        "よる",
        "おける",
        "における",
        "において",
        "いう",
        "こう",
        "そう",
        "どう",
    }
)

CONTENT_POS = frozenset({"名詞", "動詞", "形容詞"})


@dataclass(frozen=True)
class Token:
    surface: str
    lemma: str
    pos: str
    start: int
    end: int
    keep: bool


def is_content_word(pos: str, lemma: str) -> bool:
    if not lemma or lemma == "*":
        return False
    if lemma in STOPWORDS:
        return False
    if len(lemma) == 1 and "ぁ" <= lemma <= "ん":
        return False

    parts = pos.split(",")
    head = parts[0]
    sub = parts[1] if len(parts) > 1 else "*"
    if head not in CONTENT_POS:
        return False
    if head == "名詞" and sub in {"非自立", "代名詞", "接尾", "特殊"}:
        return False
    if head == "動詞" and sub != "自立":
        return False
    if head == "形容詞" and sub != "自立":
        return False
    return True


class JapaneseTokenizer:
    def __init__(self) -> None:
        try:
            from janome.tokenizer import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "Janome が入っていません。次を実行してください: py -m pip install -r requirements.txt"
            ) from exc
        self._tokenizer = Tokenizer()

    def analyze(self, text: str) -> list[Token]:
        prepared = prepare_text(text)
        tokens: list[Token] = []
        cursor = 0
        for raw in self._tokenizer.tokenize(prepared):
            surface = raw.surface
            if not surface:
                continue
            if prepared.startswith(surface, cursor):
                start = cursor
            else:
                start = prepared.find(surface, cursor)
                if start < 0:
                    continue
            end = start + len(surface)
            cursor = end
            lemma = raw.base_form
            if not lemma or lemma == "*":
                lemma = surface
            pos = raw.part_of_speech
            tokens.append(
                Token(
                    surface=surface,
                    lemma=lemma,
                    pos=pos,
                    start=start,
                    end=end,
                    keep=is_content_word(pos, lemma),
                )
            )
        return tokens

    def content_tokens(self, text: str) -> list[Token]:
        return [token for token in self.analyze(text) if token.keep]
