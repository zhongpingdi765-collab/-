from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jpsearch.textutil import prepare_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "documents.json"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    body: str


def load_documents(path: Path | None = None) -> list[Document]:
    json_path = Path(path) if path is not None else DEFAULT_CORPUS
    documents = _load_json(json_path)
    corpus_dir = json_path.parent / "corpus"
    if corpus_dir.is_dir():
        for text_path in sorted(corpus_dir.glob("*.txt")):
            documents.append(_load_text_file(text_path))
    _ensure_unique_ids(documents)
    return documents


def _load_json(path: Path) -> list[Document]:
    if not path.is_file():
        raise FileNotFoundError(f"文書ファイルがありません: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    documents: list[Document] = []
    for item in raw:
        documents.append(
            Document(
                id=str(item["id"]).strip(),
                title=prepare_text(str(item["title"])),
                body=prepare_text(str(item["body"])),
            )
        )
    return documents


def _load_text_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()
    if not lines or not lines[0].strip():
        raise ValueError(f"タイトルが空です: {path}")
    title = lines[0]
    body = "\n".join(lines[1:])
    return Document(id=path.stem, title=prepare_text(title), body=prepare_text(body))


def _ensure_unique_ids(documents: list[Document]) -> None:
    seen: set[str] = set()
    for document in documents:
        if not document.id:
            raise ValueError("文書 id が空です")
        if document.id in seen:
            raise ValueError(f"文書 id が重複しています: {document.id}")
        if not document.title or not document.body:
            raise ValueError(f"タイトルか本文が空です: {document.id}")
        seen.add(document.id)
