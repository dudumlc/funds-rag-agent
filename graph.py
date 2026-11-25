"""O agente (LangGraph): o modelo decide, as tools executam, e o ciclo repete até ele responder.

    início -> call_llm --(pediu tool?)--> tools -> call_llm ... -> fim
"""
from datetime import date

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

WEEKDAYS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]

SYSTEM_PROMPT = """\
Você é um assistente que explica, em português do Brasil e em formato de conversa, as informações \
oficiais e mais recentes de fundos imobiliários (FIIs) e de outros fundos de investimento. Suas fontes \
são os documentos publicados no Fundos.NET (B3/CVM), guardados e indexados no computador do usuário.

Hoje é {weekday}, {today}.

## Como trabalhar
1. Nunca responda de memória sobre um fundo (números, datas, eventos, regras). Tudo o que afirmar \
deve vir do resultado das tools.
2. Identifique o fundo com `listar_fundos`. Se ele não estiver cadastrado e você não tiver o CNPJ, \
peça o CNPJ ao usuário. Nunca invente um CNPJ nem um ticker: só preencha o ticker se o usuário o informou.
3. Use `sincronizar_fundo` quando o usuário perguntar por novidades ou atualizações, ou quando o fundo \
ainda não tiver documentos lidos. Não sincronize a cada mensagem.
4. Para perguntas de conteúdo (por que, como, o que diz...), use `buscar_nos_documentos`. Se os trechos \
não responderem, refine a busca (outras palavras, outra categoria, um período) antes de desistir.
5. Para "o que saiu", "o último relatório" ou "o que houve neste ano", use `listar_documentos`. A busca \
por trechos não ordena por data: compare as datas de entrega antes de dizer qual é o mais recente.

## Como responder
- Cite a fonte de cada afirmação: tipo do documento, referência, data de entrega e o link.
- Diga a "idade" da informação (ex.: "o último relatório gerencial que tenho é de agosto/2026").
- Se os documentos não trazem a resposta, diga com clareza. Não complete com suposições.
- Explique termos técnicos (DY, P/VP, vacância, LTV...) em linguagem simples na primeira vez.
- Você explica informações, não recomenda comprar, vender ou manter.
- Só PDFs são lidos. Informes em XML aparecem em `listar_documentos`, mas não entram na busca. \
Tabelas em PDF podem sair bagunçadas: avise quando houver dúvida.
- Se uma tool devolver erro, conte ao usuário o que aconteceu e o que ele pode fazer.
"""


def system_prompt(today: date | None = None) -> str:
    today = today or date.today()
    return SYSTEM_PROMPT.format(weekday=WEEKDAYS[today.weekday()], today=today.strftime("%d/%m/%Y"))


def build_graph(llm=None, checkpointer=None):
    llm = llm or init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai", max_tokens=4000)
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_llm(state: MessagesState) -> dict:
        # O prompt é montado a cada chamada (com a data de hoje) e não fica no histórico.
        messages = [SystemMessage(system_prompt()), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_llm", call_llm)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", tools_condition)   # tem tool_calls? vai para "tools"; senão, fim
    builder.add_edge("tools", "call_llm")
    return builder.compile(checkpointer=checkpointer)
