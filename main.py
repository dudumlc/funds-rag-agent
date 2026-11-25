import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from graph import build_graph


def text_of(content) -> str:
    """O conteúdo de uma resposta do modelo pode ser texto ou uma lista de blocos."""
    if isinstance(content, str):
        return content
    return "".join(b if isinstance(b, str) else b.get("text", "") for b in content if isinstance(b, (str, dict)))


def answer(graph, question: str, thread_id: str, show=print) -> str:
    """Manda a pergunta ao agente, mostra as tools que ele usa e devolve a resposta final."""
    final = ""
    config = {"configurable": {"thread_id": thread_id}}
    for update in graph.stream({"messages": [("user", question)]}, config, stream_mode="updates"):
        for node, payload in update.items():
            for message in payload["messages"]:
                if isinstance(message, AIMessage) and message.tool_calls:
                    for call in message.tool_calls:
                        show(f"  → {call['name']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})")
                elif isinstance(message, AIMessage):
                    final = text_of(message.content)
                elif isinstance(message, ToolMessage) and message.status == "error":
                    show(f"  ✗ {message.name}: erro")
    return final


def main() -> None:
    load_dotenv()
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit("Preencha GOOGLE_API_KEY no arquivo .env (copie de .env.example).")

    graph = build_graph(checkpointer=MemorySaver())   # a conversa vale até fechar o programa
    print("Agente de fundos e FIIs. Digite sua pergunta ('sair' encerra).")
    print('Exemplo: "tem novidade do fundo 15.585.932/0001-10?"\n')
    while True:
        try:
            question = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"sair", "/sair", "q", "exit"}:
            break
        try:
            print("\n" + (answer(graph, question, "chat") or "(sem resposta)") + "\n")
        except Exception as exc:   # limite de uso da API, rede...: a conversa continua
            print(f"\nErro: {type(exc).__name__}: {str(exc)[:300]}\n")


if __name__ == "__main__":
    main()
