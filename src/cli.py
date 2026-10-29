"""Консольный режим.  Запуск:  python -m src.cli "Тихий ПК для бабушки до 60 000 ₽" """
import asyncio
import json
import sys

from .agent import run_agent

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def print_step(step: dict) -> None:
    t = step["type"]
    if t == "mcp":
        print(f"🔌 MCP: инструменты {', '.join(step['tools'])}")
    elif t == "retrieve":
        print(f"📚 RAG: {', '.join(step['sources']) or '—'}")
    elif t == "llm":
        print("🧠 LLM вызывает:", ", ".join(c["name"] for c in step["tool_calls"]))
    elif t == "tool":
        print(f"🔧 {step['name']}({json.dumps(step['args'], ensure_ascii=False)})")
        print("   →", step["result"][:400].replace("\n", " "))
    elif t == "finalize":
        print(f"🏁 финализация: {step['reason']}")


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Запрос: ")
    answer, steps = asyncio.run(run_agent(question, on_step=print_step))
    print(f"\n===== Ответ агента ({len(steps)} шагов) =====\n{answer}")


if __name__ == "__main__":
    main()
