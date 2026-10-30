"""Веб-интерфейс агента.  Запуск:  streamlit run app.py"""
import asyncio
import json
import sys
import traceback

import streamlit as st

from src.agent import run_agent
from src.config import BUILDS_DIR, LLM_MODEL, MAX_STEPS

if sys.platform == "win32":  # подпроцессы asyncio на Windows
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(page_title="Агент «Сборщик ПК»", page_icon="🖥️", layout="wide")

EXAMPLES = [
    "Тихий компьютер для бабушки: смотреть сериалы и звонить внукам по видео. Бюджет 60 000 ₽",
    "ПК для монтажа видео 4K, бюджет 150 000 ₽",
    "Игровой ПК для Full HD до 90 000 ₽",
    "Хочу ПК с RTX 4090 за 50 000 ₽",
]
ICONS = {"retrieve": "📚", "llm": "🧠", "tool": "🔧", "finalize": "🏁", "mcp": "🔌"}


def root_causes(e: BaseException):
    """Разворачивает ExceptionGroup (anyio/MCP) до настоящих исключений."""
    if isinstance(e, BaseExceptionGroup):
        for sub in e.exceptions:
            yield from root_causes(sub)
    else:
        yield e


def pretty(text: str, limit: int = 1800) -> str:
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=1)
    except (json.JSONDecodeError, TypeError):
        pass
    return text if len(text) <= limit else text[:limit] + "\n…"


def render_step(step: dict, box) -> None:
    icon = ICONS.get(step["type"], "•")
    if step["type"] == "mcp":
        box.markdown(f"{icon} **MCP:** подключены инструменты: " + ", ".join(f"`{t}`" for t in step["tools"]))
    elif step["type"] == "retrieve":
        src = ", ".join(f"`{s}`" for s in step["sources"]) or "—"
        box.markdown(f"{icon} **RAG:** найдено фрагментов: {len(step['sources'])} — {src}")
    elif step["type"] == "llm":
        calls = ", ".join(f"`{c['name']}`" for c in step["tool_calls"])
        box.markdown(f"{icon} **LLM** решила вызвать: {calls}")
    elif step["type"] == "tool":
        box.markdown(f"{icon} **{step['name']}**  \nаргументы: `{json.dumps(step['args'], ensure_ascii=False)}`")
        box.code(pretty(step["result"]), language="json")
    elif step["type"] == "finalize":
        box.markdown(f"{icon} Итоговый ответ ({step['reason']})")


def render_steps(steps: list[dict]) -> None:
    with st.expander(f"Ход работы агента — {len(steps)} шагов"):
        for s in steps:
            render_step(s, st)


# ---------------------------------------------------------------- боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    model = st.text_input("Модель LLM (Ollama)", LLM_MODEL)
    st.caption(f"Предел шагов агента: {MAX_STEPS}")
    st.subheader("Примеры запросов")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    if st.button("🗑️ Очистить диалог", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    builds = sorted(BUILDS_DIR.glob("*.md"), reverse=True)[:5] if BUILDS_DIR.exists() else []
    if builds:
        st.subheader("💾 Сохранённые сборки")
        for b in builds:
            with st.expander(b.stem):
                st.markdown(b.read_text(encoding="utf-8"))

# ---------------------------------------------------------------- чат
st.title("🖥️ ИИ-агент «Сборщик ПК»")
st.caption("Подбор и проверка совместимости конфигурации под задачу и бюджет · "
           "локальная LLM + RAG (Chroma) + MCP-инструменты")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            render_steps(msg["steps"])

prompt = st.chat_input("Для чего нужен компьютер и какой бюджет?") or st.session_state.pop("pending", None)
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    llm_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history[-6:]]

    with st.chat_message("assistant"):
        status = st.status("Агент работает…", expanded=True)
        answer, steps = "", []
        try:
            answer, steps = asyncio.run(
                run_agent(prompt, llm_history, on_step=lambda s: render_step(s, status), model=model)
            )
            status.update(label=f"Готово: {len(steps)} шагов", state="complete", expanded=False)
        except Exception as e:
            traceback.print_exc()  # полный traceback в консоль, где запущен Streamlit
            status.update(label="Ошибка", state="error")
            causes = "; ".join(f"{type(c).__name__}: {c}" for c in root_causes(e))
            answer = f"⚠️ Ошибка: {causes}"
            steps = []
        st.markdown(answer)

    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": answer, "steps": steps})
    st.rerun()
