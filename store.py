"""Onde os dados ficam: SQLite (fundos e documentos) + Chroma (trechos e embeddings).

SQLite guarda QUAIS documentos existem e em que etapa estão. Chroma guarda o TEXTO dos
documentos lidos, já dividido em trechos, junto com os embeddings para a busca por significado.
Este é o único arquivo que sabe disso: para trocar por um banco na nuvem (Azure, por exemplo),
é aqui que se mexe.

Etapas de um documento (`documents.status`):
    novo      precisa ser baixado e lido
    lido      baixado, dividido em trechos e gravado no Chroma
    ignorado  só registrado (tipo que não lemos, XML, PDF sem texto...)
"""
import os
import re
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

_sleep = time.sleep   # os testes trocam por uma função vazia


def data_dir() -> Path:
    return Path(os.environ.get("FUNDOS_DATA_DIR") or Path(__file__).parent / "data")


def embedding_model() -> str:
    return os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")


# --------------------------------------------------------------------------
# Textos e CNPJ
# --------------------------------------------------------------------------
def norm(text: Optional[str]) -> str:
    """Minúsculas e sem acento, para comparar nomes e categorias."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower().strip()


def clean_cnpj(text: str) -> Optional[str]:
    """Tira a pontuação. Devolve o CNPJ (numérico ou alfanumérico) ou None se não parecer um."""
    cnpj = re.sub(r"[.\-/\s]", "", text or "").upper()
    return cnpj if re.fullmatch(r"[0-9A-Z]{12}[0-9]{2}", cnpj) else None


def date_int(iso: Optional[str]) -> int:
    """'2026-09-14T14:44' -> 20260914 (o Chroma só filtra números por faixa)."""
    try:
        return int((iso or "")[:10].replace("-", ""))
    except ValueError:
        return 0


# --------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS funds (
    cnpj   TEXT PRIMARY KEY,
    ticker TEXT,
    name   TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,      -- id do documento no FNET
    cnpj         TEXT NOT NULL,
    fund_name    TEXT,
    category     TEXT,
    doc_type     TEXT,
    reference    TEXT,
    delivered_at TEXT,                     -- ISO, ex.: 2026-09-14T14:44
    version      INTEGER,
    status       TEXT NOT NULL,            -- novo | lido | ignorado
    attempts     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    path         TEXT                      -- PDF/XML salvo em disco
);
"""


@contextmanager
def connect():
    """Abre o banco (criando pasta e tabelas se preciso); faz commit ao sair sem erro."""
    path = data_dir() / "fundos.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_fund(cnpj: str, name: Optional[str] = None, ticker: Optional[str] = None) -> None:
    ticker = (ticker or "").strip().upper() or None
    with connect() as c:
        if ticker:
            other = c.execute("SELECT cnpj FROM funds WHERE ticker = ? AND cnpj != ?", (ticker, cnpj)).fetchone()
            if other:
                raise ValueError(f"O ticker {ticker} já está cadastrado para o CNPJ {other['cnpj']}")
        c.execute(
            """INSERT INTO funds (cnpj, ticker, name) VALUES (?, ?, ?)
               ON CONFLICT(cnpj) DO UPDATE SET ticker = COALESCE(excluded.ticker, ticker),
                                               name = COALESCE(excluded.name, name)""",
            (cnpj, ticker, name or None),
        )


def find_fund(ref: str) -> str:
    """CNPJ, ticker ou parte do nome de um fundo cadastrado -> CNPJ."""
    cnpj = clean_cnpj(ref)
    if cnpj:
        return cnpj
    q = norm(ref)
    if not q:
        raise LookupError("Informe o CNPJ ou o ticker do fundo.")
    with connect() as c:
        funds = c.execute("SELECT * FROM funds").fetchall()
    hits = [f for f in funds if norm(f["ticker"]) == q] or [
        f for f in funds if q in norm(f["name"]) or q in norm(f["ticker"])
    ]
    if len(hits) == 1:
        return hits[0]["cnpj"]
    if hits:
        raise LookupError(
            f"'{ref}' corresponde a mais de um fundo: "
            + ", ".join(f"{f['ticker'] or f['name']} ({f['cnpj']})" for f in hits) + ". Informe o CNPJ."
        )
    known = ", ".join(f"{f['ticker'] or f['name']} ({f['cnpj']})" for f in funds) or "nenhum"
    raise LookupError(
        f"Não encontrei o fundo '{ref}'. Fundos cadastrados: {known}. "
        f"Para cadastrar um novo, use sincronizar_fundo com o CNPJ (e o ticker)."
    )


def get_fund(cnpj: str) -> Optional[sqlite3.Row]:
    with connect() as c:
        return c.execute("SELECT * FROM funds WHERE cnpj = ?", (cnpj,)).fetchone()


def list_funds() -> list[dict]:
    with connect() as c:
        out = []
        for f in c.execute("SELECT * FROM funds ORDER BY COALESCE(ticker, name)").fetchall():
            counts = {r["status"]: r["n"] for r in c.execute(
                "SELECT status, COUNT(*) AS n FROM documents WHERE cnpj = ? GROUP BY status", (f["cnpj"],))}
            last = c.execute("SELECT MAX(delivered_at) AS d FROM documents WHERE cnpj = ?", (f["cnpj"],)).fetchone()
            out.append({**dict(f), "documents": counts, "latest_delivered_at": last["d"]})
        return out


def count_documents(cnpj: str) -> int:
    with connect() as c:
        return c.execute("SELECT COUNT(*) AS n FROM documents WHERE cnpj = ?", (cnpj,)).fetchone()["n"]


def add_new_documents(cnpj: str, docs: list[dict], wanted) -> list[dict]:
    """Registra os documentos que ainda não conhecemos. `wanted(doc)` decide se será lido. Devolve os novos."""
    with connect() as c:
        known = {r["id"] for r in c.execute("SELECT id FROM documents WHERE cnpj = ?", (cnpj,))}
        new = [d for d in docs if d["id"] not in known]
        c.executemany(
            """INSERT OR IGNORE INTO documents
               (id, cnpj, fund_name, category, doc_type, reference, delivered_at, version, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(d["id"], cnpj, d["fund_name"], d["category"], d["doc_type"], d["reference"],
              d["delivered_at"], d["version"], "novo" if wanted(d) else "ignorado") for d in new],
        )
    return new


def pending(cnpj: str, limit: Optional[int] = None) -> list[dict]:
    """Documentos a ler. Os que já falharam vão para o fim da fila, para não travá-la."""
    sql = ("SELECT * FROM documents WHERE cnpj = ? AND status = 'novo' "
           "ORDER BY attempts, delivered_at DESC, id DESC")
    params: list = [cnpj]
    if limit is not None:
        sql, params = sql + " LIMIT ?", params + [limit]
    with connect() as c:
        return [dict(r) for r in c.execute(sql, params)]


def set_status(doc_id: int, status: str, error: Optional[str] = None, path: Optional[str] = None) -> None:
    with connect() as c:
        c.execute("UPDATE documents SET status = ?, error = ?, path = COALESCE(?, path) WHERE id = ?",
                  (status, error, path, doc_id))


def fail(doc_id: int, error: str) -> None:
    """Falha temporária (rede, API): conta a tentativa e deixa o documento na fila."""
    with connect() as c:
        c.execute("UPDATE documents SET attempts = attempts + 1, error = ? WHERE id = ?", (error[:500], doc_id))


def categories(cnpj: str) -> list[str]:
    with connect() as c:
        rows = c.execute("SELECT DISTINCT category FROM documents WHERE cnpj = ?", (cnpj,))
        return sorted(r["category"] for r in rows if r["category"])


def match_categories(cnpj: str, text: str) -> list[str]:
    """Categorias existentes cujo nome contém o texto (sem acento, sem maiúsculas)."""
    return [c for c in categories(cnpj) if norm(text) in norm(c)]


def query_documents(cnpj: str, cats: Optional[list[str]] = None, since: Optional[str] = None,
                    limit: int = 15) -> list[dict]:
    sql, params = "SELECT * FROM documents WHERE cnpj = ?", [cnpj]
    if cats:
        sql += f" AND category IN ({','.join('?' * len(cats))})"
        params += cats
    if since:
        sql += " AND delivered_at >= ?"
        params.append(since)
    sql += " ORDER BY delivered_at DESC, id DESC LIMIT ?"
    params.append(limit)
    with connect() as c:
        return [dict(r) for r in c.execute(sql, params)]


# --------------------------------------------------------------------------
# Embeddings (Gemini)
# --------------------------------------------------------------------------
_embedder = None


def set_embedder(embedder) -> None:
    """Troca o gerador de embeddings (usado nos testes). Precisa de embed_documents e embed_query."""
    global _embedder
    _embedder = embedder


def embeddings_ready() -> bool:
    return _embedder is not None or bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def get_embedder():
    global _embedder
    if _embedder is None:
        if not embeddings_ready():
            raise RuntimeError("Chave da API do Gemini não encontrada. Preencha GOOGLE_API_KEY no arquivo .env.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _embedder = GoogleGenerativeAIEmbeddings(model=embedding_model())
    return _embedder


def _retry(fn, tries: int = 4):
    """Novas tentativas com espera crescente (limite de uso da API é comum)."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception:
            if attempt == tries:
                raise
            _sleep(4 * 3 ** (attempt - 1))


def embed_documents(texts: list[str]) -> list[list[float]]:
    embedder, vectors = get_embedder(), []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        vectors += _retry(lambda b=batch: embedder.embed_documents(b))
    return vectors


# --------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------
_clients: dict = {}


def _collection():
    path = str(data_dir() / "chroma")
    if path not in _clients:
        _clients[path] = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
    return _clients[path].get_or_create_collection("fnet_documents", metadata={"hnsw:space": "cosine"})


def add_chunks(doc: dict, chunks: list[dict]) -> int:
    """Grava os trechos de um documento (substitui os que ele já tinha). chunks: [{'text', 'page'}]."""
    texts = [c["text"] for c in chunks]
    vectors = embed_documents(texts)          # primeiro os embeddings: se falhar, nada foi apagado
    col = _collection()
    col.delete(where={"doc_id": int(doc["id"])})
    col.add(
        ids=[f"{doc['id']}:{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=vectors,
        metadatas=[{
            "doc_id": int(doc["id"]), "cnpj": doc["cnpj"], "fund_name": doc["fund_name"] or "",
            "category": doc["category"] or "", "doc_type": doc["doc_type"] or "",
            "reference": doc["reference"] or "", "delivered_at": doc["delivered_at"] or "",
            "delivered_int": date_int(doc["delivered_at"]), "page": int(c["page"]),
        } for c in chunks],
    )
    return len(chunks)


def count_chunks(cnpj: Optional[str] = None) -> int:
    col = _collection()
    return col.count() if cnpj is None else len(col.get(where={"cnpj": cnpj}, include=[])["ids"])


def search(question: str, cnpj: Optional[str] = None, cats: Optional[list[str]] = None,
           since: Optional[str] = None, k: int = 6) -> list[dict]:
    """Trechos mais parecidos com a pergunta, com filtros opcionais. O mais parecido vem primeiro."""
    col = _collection()
    if col.count() == 0:
        return []
    conditions = []
    if cnpj:
        conditions.append({"cnpj": cnpj})
    if cats:
        conditions.append({"category": {"$in": cats}})
    if since:
        conditions.append({"delivered_int": {"$gte": date_int(since)}})
    where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}

    query_vector = _retry(lambda: get_embedder().embed_query(question))
    result = col.query(query_embeddings=[query_vector], n_results=k, where=where,
                       include=["documents", "metadatas", "distances"])
    return [
        {"text": text, "meta": meta, "similarity": round(1 - float(dist), 4)}
        for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]
