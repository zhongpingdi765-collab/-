from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jpsearch.corpus import DEFAULT_CORPUS
from jpsearch.engine import SearchEngine
from jpsearch.web import serve


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="転置インデックスと BM25 による日本語文書検索")
    parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS, help="文書 JSON のパス")
    sub = parser.add_subparsers(dest="command")

    serve_cmd = sub.add_parser("serve", help="ブラウザで検索する")
    serve_cmd.add_argument("--port", type=int, default=8765)
    serve_cmd.add_argument("--no-browser", action="store_true")

    search_cmd = sub.add_parser("search", help="クエリに近い文書を表示する")
    search_cmd.add_argument("query", nargs="+")
    search_cmd.add_argument("-k", type=int, default=5)

    analyze_cmd = sub.add_parser("analyze", help="クエリがどの語として索引に入るか表示する")
    analyze_cmd.add_argument("query", nargs="+")

    postings_cmd = sub.add_parser("postings", help="1語の転置インデックスを表示する")
    postings_cmd.add_argument("word")

    sub.add_parser("stats", help="索引の統計を表示する")

    args = parser.parse_args(argv)
    command = args.command or "serve"
    if command == "serve":
        print("形態素解析の辞書を読み、索引を作っています...", flush=True)
    engine = SearchEngine.load(args.data)

    if command == "serve":
        port = getattr(args, "port", 8765)
        open_browser = not getattr(args, "no_browser", False)
        serve(engine, port=port, open_browser=open_browser)
        return 0
    if command == "search":
        _print_search(engine, " ".join(args.query), args.k)
        return 0
    if command == "analyze":
        _print_analyze(engine, " ".join(args.query))
        return 0
    if command == "postings":
        return _print_postings(engine, args.word)
    if command == "stats":
        _print_stats(engine)
        return 0
    parser.print_help()
    return 1


def _print_search(engine: SearchEngine, query: str, k: int) -> None:
    response = engine.search(query, k=k)
    if response.terms:
        print("検索語: " + " / ".join(response.terms))
    else:
        print("検索に使える語がありません。名詞や動詞を含めてください。")
        return
    if response.dropped:
        print("除外: " + " / ".join(response.dropped))
    if not response.results:
        print("一致する文書はありません。")
        return
    print()
    for rank, result in enumerate(response.results, start=1):
        print(f"{rank}. {result.score:.3f}  {result.title}  [{result.doc_id}]")
        for item in result.contributions:
            print(
                f"   {item.term}  文書内{item.tf}回  出現文書{item.df}  "
                f"idf={item.idf:.3f}  寄与={item.score:.3f}"
            )
        print(f"   {result.snippet_text}")
        print()


def _print_analyze(engine: SearchEngine, query: str) -> None:
    for token in engine.tokenizer.analyze(query):
        pos = ",".join(token.pos.split(",")[:2])
        mark = "使う" if token.keep else "捨てる"
        print(f"[{mark}] {token.surface} → {token.lemma}（{pos}）")


def _print_postings(engine: SearchEngine, word: str) -> int:
    found = engine.postings(word)
    if found is None:
        print(f"索引にありません: {word}")
        return 1
    lemma, rows = found
    print(f"{lemma}  出現文書数 {len(rows)}")
    for document, tf in rows:
        print(f"  tf={tf:<3} {document.title}  [{document.id}]")
    return 0


def _print_stats(engine: SearchEngine) -> None:
    stats = engine.stats()
    print(f"文書数: {stats['n_docs']}")
    print(f"語彙数: {stats['vocabulary']}")
    print(f"平均文書長: {stats['avgdl']} 語")
    print(f"BM25 パラメータ: k1={stats['k1']}  b={stats['b']}")
    print()
    for item in stats["documents"]:
        print(f"  {item['id']:<16} {item['title']}")


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
