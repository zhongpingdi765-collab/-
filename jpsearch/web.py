from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from jpsearch.engine import SearchEngine

TEMPLATE = Path(__file__).resolve().parent / "templates" / "index.html"
_LOCK = threading.Lock()


class SearchHandler(BaseHTTPRequestHandler):
    engine: SearchEngine

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, TEMPLATE.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/stats":
            self._send_json(self.engine.stats())
            return
        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            try:
                limit = int(params.get("k", ["8"])[0])
            except ValueError:
                limit = 8
            with _LOCK:
                payload = self.engine.search(query, k=min(max(limit, 1), 20))
            self._send_json(payload.as_dict())
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        if args and str(args[0]).startswith("GET /api/"):
            return
        super().log_message(fmt, *args)

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(engine: SearchEngine, port: int = 8765, open_browser: bool = True) -> None:
    SearchHandler.engine = engine
    stats = engine.stats()
    server = ThreadingHTTPServer(("127.0.0.1", port), SearchHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"文書数 {stats['n_docs']} / 語彙数 {stats['vocabulary']} / 平均文書長 {stats['avgdl']}", flush=True)
    print(f"検索画面: {url}", flush=True)
    print("止めるときは Ctrl+C", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました")
    finally:
        server.server_close()
