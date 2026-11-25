"""As ferramentas (tools) do agente. Cada uma é uma função; o modelo escolhe qual chamar.

Regra de todas: nunca levantam erro para o modelo. Se algo der errado, devolvem um JSON com
status "erro" e uma mensagem em português, e o modelo explica ao usuário (ou tenta de novo).

A sincronização (baixar -> ler o PDF -> dividir em trechos -> gravar no Chroma) também mora aqui.
"""
import functools
import json
import re
import time
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import store
from fnet import Fnet, link

# Só estes documentos são baixados e lidos (comparação sem acento, pelo INÍCIO do nome).
# O resto fica apenas listado. Se um tipo importante de FII não estiver sendo lido, ajuste aqui.
LER_CATEGORIAS = ("regulamento", "fato relevante", "comunicado", "assembleia", "atos de", "aviso", "relatorio")
LER_TIPOS = ("demonstracoes", "relatorio")

CHUNK_SIZE = 1200      # caracteres por trecho
CHUNK_OVERLAP = 150    # quanto um trecho repete do anterior
MIN_CHUNK = 40         # descarta trechos minúsculos (rodapé, número de página)
PAUSE = 0.5            # segundos entre downloads, para não sobrecarregar o site
MAX_STREAK = 3         # erros seguidos que interrompem uma sincronização

SITUACAO = {"novo": "na fila para leitura", "lido": "lido e pesquisável", "ignorado": "só registrado (não é lido)"}


# --------------------------------------------------------------------------
# Apoio
# --------------------------------------------------------------------------
def safe(fn):
    """Faz a tool devolver sempre um texto JSON, mesmo quando dá erro."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            known = isinstance(exc, (LookupError, ValueError, RuntimeError))
            result = {"status": "erro", "message": str(exc) if known else f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False)

    return wrapper


def wants(doc: dict) -> bool:
    return store.norm(doc["category"]).startswith(LER_CATEGORIAS) or store.norm(doc["doc_type"]).startswith(LER_TIPOS)


def br(iso: str) -> str:
    try:
        return datetime.fromisoformat((iso or "")[:19]).strftime("%d/%m/%Y")
    except ValueError:
        return iso or "?"


def to_iso(text: str) -> str:
    """DD/MM/AAAA, MM/AAAA, AAAA-MM-DD ou AAAA-MM -> 'AAAA-MM-DD'."""
    text = text.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%Y", "%Y-%m"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Data inválida: '{text}'. Use DD/MM/AAAA, MM/AAAA ou AAAA-MM-DD.")


def label(doc: dict) -> str:
    return doc["category"] + (f" - {doc['doc_type']}" if doc["doc_type"] else "")


def brief(doc: dict) -> dict:
    return {"id": doc["id"], "documento": label(doc), "referencia": doc["reference"],
            "entregue_em": br(doc["delivered_at"]), "link": link(doc["id"])}


# --------------------------------------------------------------------------
# PDF -> trechos
# --------------------------------------------------------------------------
class Unreadable(Exception):
    """PDF que não adianta tentar de novo (protegido, corrompido ou sem texto)."""


def pdf_chunks(doc: dict, path: Path) -> list[dict]:
    """Texto do PDF dividido em trechos: [{'text': ..., 'page': ...}]. Cada trecho começa com uma
    linha dizendo de onde veio (fundo, documento, data, página), o que ajuda a busca."""
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise Unreadable("PDF protegido por senha")
        pages = [(n, page.extract_text() or "") for n, page in enumerate(reader.pages, start=1)]
    except Unreadable:
        raise
    except Exception as exc:
        raise Unreadable(f"Não consegui ler o PDF: {type(exc).__name__}: {exc}") from exc

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for number, text in pages:
        text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text.replace("\x00", " "))).strip()
        header = (f"[{doc['fund_name'] or doc['cnpj']} | {label(doc)} | referência {doc['reference'] or '?'} | "
                  f"entregue em {br(doc['delivered_at'])} | pág. {number}]")
        for part in splitter.split_text(text):
            if len(part.strip()) >= MIN_CHUNK:
                chunks.append({"text": f"{header}\n{part.strip()}", "page": number})
    if not chunks:
        raise Unreadable("PDF sem texto (provavelmente escaneado)")
    return chunks


# --------------------------------------------------------------------------
# Sincronização
# --------------------------------------------------------------------------
def sync(fund: str, ticker: str = "", max_downloads: int = 5) -> dict:
    cnpj = store.find_fund(fund)
    first_time = store.count_documents(cnpj) == 0

    client = Fnet(cnpj)
    docs = client.list_documents(pages=3 if first_time else 1)    # na 1ª vez traz mais histórico
    if not docs and store.get_fund(cnpj) is None:
        return {"status": "sem_documentos", "message": f"O Fundos.NET não retornou documentos para o CNPJ {cnpj}. Confira o número."}

    store.save_fund(cnpj, name=docs[0]["fund_name"] if docs else None, ticker=ticker)
    new = store.add_new_documents(cnpj, docs, wanted=wants)

    todo = store.pending(cnpj, limit=max(0, max_downloads))
    if todo:
        store.get_embedder()        # falha já aqui, com mensagem clara, se faltar a chave da API

    read, ignored, errors, streak = [], 0, [], 0
    for doc in todo:
        try:
            path = Path(doc["path"]) if doc["path"] else None
            if path is None or not path.exists():          # já baixado antes? reaproveita o arquivo
                data, ext = client.download(doc["id"])
                path = Path(store.data_dir()) / "files" / cnpj / f"{(doc['delivered_at'] or 'sem-data')[:10]}_{doc['id']}.{ext}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                store.set_status(doc["id"], "novo", path=str(path))
                time.sleep(PAUSE)
                if ext != "pdf":
                    store.set_status(doc["id"], "ignorado", f"formato {ext.upper()} não é lido")
                    ignored += 1
                    continue
            store.add_chunks(doc, pdf_chunks(doc, path))
            store.set_status(doc["id"], "lido")
            read.append(brief(doc))
            streak = 0
        except Unreadable as exc:
            store.set_status(doc["id"], "ignorado", str(exc))
            ignored += 1
        except Exception as exc:
            store.fail(doc["id"], f"{type(exc).__name__}: {exc}")
            errors.append(f"id {doc['id']} ({label(doc)}): {type(exc).__name__}: {exc}")
            streak += 1
            if streak >= MAX_STREAK:
                errors.append("Interrompido após falhas seguidas; o restante fica para a próxima sincronização.")
                break

    fund_row = store.get_fund(cnpj)
    name = fund_row["ticker"] or fund_row["name"] or cnpj
    left = len(store.pending(cnpj))
    latest = brief(docs[0]) if docs else None
    parts = [f"{len(new)} documento(s) novo(s) de {name}." if new else f"Nenhum documento novo de {name}."]
    if latest:
        parts.append(f"O mais recente é {latest['documento']} (ref. {latest['referencia']}), entregue em {latest['entregue_em']}.")
    if read:
        parts.append(f"Li e indexei {len(read)} documento(s).")
    if left:
        parts.append(f"Ainda há {left} na fila, que serão lidos nas próximas sincronizações.")
    if errors:
        parts.append(f"Houve {len(errors)} problema(s); veja 'erros'.")
    return {
        "status": "ok", "message": " ".join(parts),
        "fundo": {"nome": fund_row["name"], "ticker": fund_row["ticker"], "cnpj": cnpj},
        "novos": len(new), "novos_mais_recentes": [brief(d) for d in new[:5]],
        "documento_mais_recente": latest,
        "lidos_agora": read, "ignorados_agora": ignored, "fila_restante": left, "erros": errors,
    }


# --------------------------------------------------------------------------
# As tools
# --------------------------------------------------------------------------
@tool(parse_docstring=True)
@safe
def sincronizar_fundo(fundo: str, ticker: str = "", max_downloads: int = 5) -> dict:
    """Busca no Fundos.NET (B3) os documentos de um fundo, registra os novos e lê os PDFs mais recentes
    para que possam ser pesquisados. Acessa a internet e pode demorar. Use quando o usuário pedir
    novidades ou atualização de um fundo, ou quando o fundo ainda não tiver documentos lidos.

    Args:
        fundo: CNPJ do fundo (com ou sem pontuação) ou ticker/nome de um fundo já cadastrado.
        ticker: Ticker ou apelido para cadastrar junto com o CNPJ. Preencha SOMENTE se o usuário informou; nunca invente.
        max_downloads: Máximo de documentos a ler nesta chamada (padrão 5). O que sobrar fica para a próxima.
    """
    return sync(fundo, ticker, max_downloads)


@tool(parse_docstring=True)
@safe
def listar_fundos() -> dict:
    """Lista os fundos cadastrados, com a situação dos documentos de cada um. Use para saber
    quais fundos o agente conhece antes de responder sobre um fundo.
    """
    funds = [
        {"ticker": f["ticker"], "nome": f["name"], "cnpj": f["cnpj"],
         "documento_mais_recente_entregue_em": br(f["latest_delivered_at"]) if f["latest_delivered_at"] else None,
         "documentos_por_situacao": {SITUACAO.get(k, k): v for k, v in f["documents"].items()}}
        for f in store.list_funds()
    ]
    return {"status": "ok", "fundos": funds,
            "message": f"{len(funds)} fundo(s) cadastrado(s)." if funds else "Nenhum fundo cadastrado ainda."}


@tool(parse_docstring=True)
@safe
def listar_documentos(fundo: str, categoria: str = "", desde: str = "", limite: int = 15) -> dict:
    """Lista os documentos registrados de um fundo (de todos os tipos), do mais recente para o mais antigo.
    Use para saber o que saiu, qual é o último relatório ou o que houve num período. Não acessa a internet.

    Args:
        fundo: CNPJ, ticker ou nome do fundo cadastrado.
        categoria: Filtro opcional pelo nome da categoria (ex.: "fato relevante", "relatório").
        desde: Data inicial opcional (DD/MM/AAAA, MM/AAAA ou AAAA-MM-DD).
        limite: Quantidade máxima de documentos (padrão 15, máximo 50).
    """
    cnpj = store.find_fund(fundo)
    cats = None
    if categoria.strip():
        cats = store.match_categories(cnpj, categoria)
        if not cats:
            return {"status": "sem_resultados", "message": f"Nenhuma categoria parecida com '{categoria}'.",
                    "categorias_disponiveis": store.categories(cnpj)}
    rows = store.query_documents(cnpj, cats, to_iso(desde) if desde.strip() else None, max(1, min(int(limite), 50)))
    docs = [
        {"id": r["id"], "categoria": r["category"], "tipo": r["doc_type"], "referencia": r["reference"],
         "entregue_em": br(r["delivered_at"]),
         "situacao": SITUACAO.get(r["status"], r["status"]) + (f" ({r['error']})" if r["status"] == "ignorado" and r["error"] else ""),
         "link": link(r["id"])}
        for r in rows
    ]
    return {"status": "ok" if docs else "sem_resultados",
            "message": f"{len(docs)} documento(s), do mais recente para o mais antigo." if docs else "Nenhum documento com esses filtros.",
            "documentos": docs}


@tool(parse_docstring=True)
@safe
def buscar_nos_documentos(pergunta: str, fundo: str = "", categoria: str = "", desde: str = "", n_resultados: int = 6) -> dict:
    """Busca por significado nos PDFs já lidos e devolve os trechos mais parecidos com a pergunta, com
    documento, data, página e link. Use para perguntas de conteúdo (por que, como, o que diz...).
    Os trechos vêm por semelhança, NÃO por data: confira as datas antes de dizer o que é mais recente.

    Args:
        pergunta: O assunto ou a pergunta, em linguagem natural.
        fundo: CNPJ, ticker ou nome do fundo. Vazio busca em todos os fundos.
        categoria: Filtro opcional pelo nome da categoria (ex.: "regulamento").
        desde: Data inicial opcional (DD/MM/AAAA, MM/AAAA ou AAAA-MM-DD).
        n_resultados: Quantos trechos devolver (padrão 6, máximo 10).
    """
    if not pergunta.strip():
        raise ValueError("Informe a pergunta ou o assunto da busca.")
    cnpj = store.find_fund(fundo) if fundo.strip() else None
    cats = None
    if categoria.strip():
        if not cnpj:
            raise ValueError("Para filtrar por categoria, informe também o fundo.")
        cats = store.match_categories(cnpj, categoria)
        if not cats:
            return {"status": "sem_resultados", "message": f"Nenhuma categoria parecida com '{categoria}'.",
                    "categorias_disponiveis": store.categories(cnpj)}
    if store.count_chunks(cnpj) == 0:
        return {"status": "sem_indice", "message": "Ainda não há documentos lidos para isso. Use sincronizar_fundo primeiro."}

    hits = store.search(pergunta, cnpj, cats, to_iso(desde) if desde.strip() else None, max(1, min(int(n_resultados), 10)))
    if not hits:
        return {"status": "sem_resultados", "message": "Nenhum trecho encontrado com esses filtros."}
    return {
        "status": "ok",
        "message": f"{len(hits)} trecho(s), do mais ao menos parecido com a pergunta (não é ordem de data).",
        "resultados": [
            {"similaridade": h["similarity"], "fundo": h["meta"]["fund_name"], "documento": h["meta"]["category"]
             + (f" - {h['meta']['doc_type']}" if h["meta"]["doc_type"] else ""),
             "referencia": h["meta"]["reference"], "entregue_em": br(h["meta"]["delivered_at"]),
             "pagina": h["meta"]["page"], "link": link(h["meta"]["doc_id"]), "trecho": h["text"][:1500]}
            for h in hits
        ],
    }


TOOLS = [sincronizar_fundo, listar_fundos, listar_documentos, buscar_nos_documentos]
