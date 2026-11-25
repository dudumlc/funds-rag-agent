# Agente de fundos e FIIs (versão local)

Chat no terminal que responde sobre fundos e FIIs usando os documentos oficiais do Fundos.NET (B3).
Você pergunta, o agente busca os relatórios, salva, vetoriza e responde citando documento, data e link.

## Rodar

Requer Python 3.11 ou mais novo.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# edite o .env e coloque sua GOOGLE_API_KEY (https://aistudio.google.com/apikey)

python main.py
```

Exemplos de conversa:

- `tem novidade do fundo 15.585.932/0001-10?` (o agente sincroniza sozinho; diga o ticker se quiser cadastrá-lo)
- `o que o regulamento diz sobre resgate?`
- `qual foi o último relatório gerencial?`

## Como funciona

```
pergunta ──► graph.py (Gemini decide qual tool chamar) ──► tools.py
                                                             │
        sincronizar_fundo ── fnet.py: lista e baixa PDFs ────┤
                             tools.py: PDF → trechos         │
                             store.py: grava ────────────────┤
        listar_documentos / buscar_nos_documentos ◄── store.py
```

| Arquivo | O que faz |
|---|---|
| `main.py` | O chat no terminal. |
| `graph.py` | O agente (LangGraph) e o prompt do sistema. |
| `tools.py` | As 4 tools e a sincronização (PDF → trechos). |
| `fnet.py` | Fala com o Fundos.NET: lista e baixa documentos. |
| `store.py` | Guarda tudo: SQLite (fundos e documentos) e Chroma (trechos e embeddings). |

Os dados ficam em `data/` (fora do git): `fundos.db`, `chroma/` e `files/` (os PDFs baixados).

Cada documento passa por: `novo` → `lido` (ou `ignorado`, para XML, tipos que não lemos e PDF sem texto).
Se um download ou embedding falhar, o documento continua `novo` e vai para o fim da fila.
