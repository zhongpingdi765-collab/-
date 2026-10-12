# 日本語文書検索

## 起動

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py
```

http://127.0.0.1:8765

`起動.bat` でも起動できます。

## コマンド

```powershell
.\.venv\Scripts\python main.py search 台風の被害
.\.venv\Scripts\python main.py analyze 走った
.\.venv\Scripts\python main.py postings 台風
.\.venv\Scripts\python main.py stats
```

## 文章の追加

`data/documents.json` に `id`、`title`、`body` を追加する。
`data/corpus/` にテキストを置く場合は、1行目がタイトル、以降が本文。ファイル名が id になる。追加後は再起動する。

## テスト

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```
