"""Cliente do Fundos.NET (B3): lista e baixa os documentos de um fundo pelo CNPJ.

É a mesma chamada que a página pública do FNET faz. O site é lento e às vezes instável,
por isso há timeout generoso e novas tentativas. Este arquivo só fala HTTP: quem grava
no banco ou em disco é o tools.py.
"""
import base64
import binascii
import time
from datetime import datetime

import requests

BASE = "https://fnet.bmfbovespa.com.br/fnet/publico"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
TIMEOUT = (10, 90)   # (conexão, leitura) em segundos
RETRIES = 3
BACKOFF = 5          # segundos x número da tentativa


class FnetError(RuntimeError):
    """Falha ao falar com o Fundos.NET."""


def link(doc_id: int) -> str:
    return f"{BASE}/downloadDocumento?id={doc_id}"


def _iso(raw: str) -> str:
    """'14/09/2026 14:44' -> '2026-09-14T14:44' (se não reconhecer, devolve o texto)."""
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime((raw or "").strip(), fmt).isoformat(timespec="minutes")
        except ValueError:
            pass
    return (raw or "").strip()


def _row(r: dict) -> dict:
    """Linha crua da resposta do FNET -> campos que usamos."""
    return {
        "id": int(r["id"]),
        "fund_name": (r.get("descricaoFundo") or "").strip(),
        "category": (r.get("categoriaDocumento") or "").strip(),
        "doc_type": (r.get("tipoDocumento") or "").strip(),
        "reference": (r.get("dataReferencia") or "").strip(),
        "delivered_at": _iso(r.get("dataEntrega")),
        "version": r.get("versao"),
    }


def detect_file(raw: bytes) -> tuple[bytes, str]:
    """Descobre o formato do download: PDF, XML, ou PDF/XML em base64. Devolve (bytes, extensão)."""
    head = raw.lstrip()[:100].lower()
    if raw.startswith(b"%PDF"):
        return raw, "pdf"
    if head.startswith(b"<?xml"):
        return raw, "xml"
    if head.startswith((b"<!doctype html", b"<html")):
        raise FnetError("O FNET devolveu uma página HTML em vez do arquivo (sessão expirada ou bloqueio)")
    try:
        data = base64.b64decode(raw)
    except (binascii.Error, ValueError):
        raise FnetError("Formato de arquivo desconhecido")
    if data.startswith(b"%PDF"):
        return data, "pdf"
    if data.lstrip().startswith(b"<?xml"):
        return data, "xml"
    raise FnetError("Formato de arquivo desconhecido (nem PDF nem XML)")


class Fnet:
    """Sessão HTTP com os cookies do FNET, para um fundo."""

    def __init__(self, cnpj: str, session=None):
        self.cnpj = cnpj
        self._session = session
        self._ready = session is not None   # sessão injetada (testes) já vem pronta

    def _get(self, url: str, **kwargs) -> requests.Response:
        if not self._ready:
            self._session = self._session or requests.Session()
            page = f"{BASE}/abrirGerenciadorDocumentosCVM?cnpjFundo={self.cnpj}"
            self._session.headers.update({**HEADERS, "Referer": page, "X-Requested-With": "XMLHttpRequest"})
            self._request(page)          # abrir a página do fundo entrega os cookies de sessão
            self._ready = True
        return self._request(url, **kwargs)

    def _request(self, url: str, **kwargs) -> requests.Response:
        """GET com timeout e novas tentativas."""
        last = None
        for attempt in range(1, RETRIES + 1):
            try:
                resp = self._session.get(url, timeout=TIMEOUT, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last = exc
                if attempt < RETRIES:
                    time.sleep(BACKOFF * attempt)
        raise FnetError(f"Fundos.NET não respondeu: {type(last).__name__}: {last}") from last

    def list_documents(self, pages: int = 1, page_size: int = 100) -> list[dict]:
        """Documentos do fundo, do mais recente para o mais antigo (por data de entrega)."""
        docs, seen = [], set()
        for page in range(pages):
            resp = self._get(
                f"{BASE}/pesquisarGerenciadorDocumentosDados",
                params={"d": 0, "s": page * page_size, "l": page_size,
                        "o[0][dataEntrega]": "desc", "cnpjFundo": self.cnpj},
            )
            try:
                rows = resp.json().get("data")
            except ValueError:
                rows = None
            if not isinstance(rows, list):
                raise FnetError("Resposta inesperada do FNET (não veio a lista de documentos)")
            fresh = [_row(r) for r in rows if int(r["id"]) not in seen]
            if not fresh:          # página repetida ou vazia: a paginação foi ignorada
                break
            docs += fresh
            seen.update(d["id"] for d in fresh)
            if len(rows) < page_size:
                break
        # Não dependemos de o site respeitar o pedido de ordenação: ordenamos aqui.
        return sorted(docs, key=lambda d: d["delivered_at"], reverse=True)

    def download(self, doc_id: int) -> tuple[bytes, str]:
        """Baixa um documento. Devolve (conteúdo, extensão), com extensão 'pdf' ou 'xml'."""
        return detect_file(self._get(f"{BASE}/downloadDocumento", params={"id": doc_id}).content)
