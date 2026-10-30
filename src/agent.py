"""Оркестратор агента (LangGraph) + MCP-клиент + RAG + локальная LLM (Ollama).

Граф агента:

    retrieve ──► build_prompt ──► llm ──┬──► tools ──► llm  (цикл, пока LLM вызывает инструменты)
                                        ├──► finalize ──► END  (лимит шагов или пустой ответ)
                                        └──► END              (готовый ответ)

Изменения для маленьких моделей (qwen3:4b):
  * num_predict ограничивает длину ответа, чтобы модель не зацикливалась;
  * температура 0.7 вместо 0 (при 0 маленькие модели часто повторяют одно и то же);
  * контекст не больше 6144, RAG-фрагментов не больше 3 (короче промпт, быстрее ответ);
  * в терминал пишется лог каждого запроса к LLM;
  * запасной парсер: если модель написала вызов инструмента текстом <tool_call>{...}</tool_call>,
    он превращается в настоящий вызов.
"""
import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Callable, TypedDict

import ollama
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import (
    KNOWLEDGE_COLLECTION, LLM_MODEL, MAX_STEPS, NUM_CTX, OLLAMA_HOST, ROOT, TOP_K,
)

# Настройки для слабого железа и маленьких моделей. При необходимости меняйте здесь.
CTX_USED = min(NUM_CTX, 6144)   # размер контекста
TOP_K_USED = min(TOP_K, 3)      # сколько фрагментов из базы знаний кладём в промпт
NUM_PREDICT = 1024              # максимум токенов в одном ответе модели
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.8

SYSTEM_PROMPT = """Ты — ИИ-агент «Сборщик ПК». Ты подбираешь конфигурацию компьютера под задачу и бюджет пользователя.

Порядок работы:
1. Определи задачи пользователя и бюджет. Опирайся на справочные материалы из базы знаний.
2. Подбери детали инструментом search_components по каждой нужной категории:
   cpu, motherboard, ram, storage, psu, case; gpu — только если задача требует видеокарту
   или у процессора нет встроенной графики; cooler — если процессор без кулера или нужна тишина.
   Учитывай сокет процессора и тип памяти платы через фильтры socket и mem_type.
3. Обязательно вызови check_compatibility со списком id всех деталей.
   Если compatible=false — замени проблемные детали и проверь снова.
4. Вызови calc_total с бюджетом. Если сумма больше бюджета — замени дорогие детали
   на более дешёвые и снова проверь совместимость.
5. Когда сборка совместима и укладывается в бюджет — вызови save_build.
6. Дай итоговый ответ по-русски: таблица «Категория | Модель | Цена», итоговая сумма,
   краткое обоснование выбора со ссылками на источники в квадратных скобках, например [07_okhlazhdenie_i_shum.md].

Правила:
- Используй ТОЛЬКО id и цены, полученные от инструментов. Не придумывай комплектующие.
- Все проверки совместимости и суммы делай инструментами, а не в уме.
- Не вызывай один и тот же инструмент с теми же аргументами повторно.
- Если задачу невозможно выполнить в рамках бюджета, честно объясни почему и предложи
  реалистичную альтернативу (другой бюджет или сниженные требования)."""

FINALIZE_PROMPT = (
    "Сформулируй итоговый ответ пользователю по результатам работы инструментов выше, "
    "без новых вызовов инструментов. Если сборка не завершена — честно напиши, что удалось, а что нет."
)

StepCallback = Callable[[dict], None]


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


class AgentState(TypedDict):
    question: str
    history: list[dict]
    context: list[dict]
    messages: list[Any]
    steps: list[dict]
    iterations: int
    answer: str


# ---------------------------------------------------------------- MCP -> Ollama
def _simplify_schema(schema: dict) -> dict:
    """JSON Schema из MCP -> упрощённая схема для Ollama (без anyOf/null)."""
    props = {}
    for name, prop in (schema.get("properties") or {}).items():
        p = dict(prop)
        if "anyOf" in p:
            variants = [v for v in p.pop("anyOf") if v.get("type") != "null"]
            if variants:
                p.update(variants[0])
        props[name] = {k: p[k] for k in ("type", "description", "items", "enum") if k in p}
    return {"type": "object", "properties": props, "required": schema.get("required", [])}


def mcp_tool_to_ollama(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": _simplify_schema(tool.inputSchema),
        },
    }


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _parse_text_tool_calls(text: str) -> list:
    """Запасной вариант: достаёт вызовы вида <tool_call>{"name":..., "arguments":{...}}</tool_call>
    из текста ответа, если модель не оформила их как настоящие tool_calls."""
    calls = []
    for raw in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", flags=re.S):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = data.get("name")
        if not name:
            continue
        args = data.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append(ollama.Message.ToolCall(
            function=ollama.Message.ToolCall.Function(name=name, arguments=args)))
    return calls


# ---------------------------------------------------------------- агент
class PCBuildAgent:
    def __init__(self, session: ClientSession, tools: list[dict],
                 on_step: StepCallback | None = None, model: str | None = None):
        self.session = session
        self.tools = tools
        self.schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        self.on_step = on_step
        self.model = model or LLM_MODEL
        self.llm = ollama.AsyncClient(host=OLLAMA_HOST)
        try:
            from .rag import HybridRetriever
            self.retriever = HybridRetriever(KNOWLEDGE_COLLECTION)
        except Exception as e:
            _log(f"[agent] RAG недоступен: {e}")
            self.retriever = None
        self.graph = self._build_graph()

    # --- вспомогательное
    def _step(self, state: AgentState, step: dict) -> list[dict]:
        if self.on_step:
            self.on_step(step)
        return state["steps"] + [step]

    def _fix_args(self, name: str, args: dict) -> dict:
        """Мелкие модели иногда передают список строкой или число строкой — чиним."""
        props = self.schemas.get(name, {}).get("properties", {})
        fixed = {}
        for key, value in (args or {}).items():
            if key not in props or value is None or value == "":
                continue
            typ = props[key].get("type")
            if typ == "array" and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = [v.strip(" '\"[]") for v in value.split(",") if v.strip(" '\"[]")]
            elif typ == "integer" and isinstance(value, (str, float)):
                try:
                    value = int(float(str(value).replace(" ", "")))
                except ValueError:
                    continue
            fixed[key] = value
        return fixed

    async def _chat(self, messages: list, use_tools: bool = True):
        kwargs = dict(
            model=self.model, messages=messages,
            options={"num_ctx": CTX_USED, "temperature": LLM_TEMPERATURE,
                     "top_p": LLM_TOP_P, "num_predict": NUM_PREDICT},
        )
        if use_tools:
            kwargs["tools"] = self.tools
        try:
            return await self.llm.chat(think=False, **kwargs)   # отключаем «размышления» Qwen3 ради скорости
        except (TypeError, ollama.ResponseError) as e:
            if isinstance(e, ollama.ResponseError) and "think" not in str(e).lower():
                raise
            return await self.llm.chat(**kwargs)                 # модель/клиент без поддержки think

    # --- узлы графа
    async def retrieve(self, state: AgentState) -> dict:
        context = self.retriever.search(state["question"], k=TOP_K_USED) if self.retriever else []
        step = {"type": "retrieve", "sources": [c["source"] for c in context],
                "fragments": [c["text"][:300] for c in context]}
        return {"context": context, "steps": self._step(state, step)}

    async def build_prompt(self, state: AgentState) -> dict:
        context_text = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in state["context"]) \
            or "(база знаний недоступна)"
        user_msg = (f"Справочные материалы из базы знаний:\n{context_text}\n\n"
                    f"Запрос пользователя: {state['question']}")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state["history"],
                    {"role": "user", "content": user_msg}]
        return {"messages": messages}

    async def call_llm(self, state: AgentState) -> dict:
        step_no = state["iterations"] + 1
        t0 = time.time()
        _log(f"[llm] запрос к модели {self.model}, шаг {step_no}")
        response = await self._chat(state["messages"])
        msg = response.message

        # Запасной парсер: модель написала вызов текстом, а не как tool_call
        if not msg.tool_calls and msg.content and "<tool_call>" in msg.content:
            parsed = _parse_text_tool_calls(msg.content)
            if parsed:
                _log(f"[llm] tool_call найден в тексте ответа: {len(parsed)} шт.")
                msg = ollama.Message(role="assistant", content="", tool_calls=parsed)

        _log(f"[llm] ответ за {time.time() - t0:.1f} c; tool_calls={len(msg.tool_calls or [])}; "
             f"текст: {(msg.content or '')[:200]!r}")

        calls = [{"name": tc.function.name, "args": dict(tc.function.arguments or {})}
                 for tc in (msg.tool_calls or [])]
        update = {"messages": state["messages"] + [msg], "iterations": step_no,
                  "answer": _strip_think(msg.content)}
        if calls:
            update["steps"] = self._step(state, {"type": "llm", "tool_calls": calls,
                                                 "text": _strip_think(msg.content)})
        return update

    async def call_tools(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        new_messages, steps = [], state["steps"]
        for tc in last.tool_calls or []:
            name = tc.function.name
            args = self._fix_args(name, dict(tc.function.arguments or {}))
            _log(f"[tool] {name} {args}")
            try:
                result = await asyncio.wait_for(self.session.call_tool(name, args), timeout=300)
                _log(f"[tool] {name}: готово")
                text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
                if result.isError:
                    text = json.dumps({"error": text}, ensure_ascii=False)
            except Exception as e:
                text = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
            step = {"type": "tool", "name": name, "args": args, "result": text}
            if self.on_step:
                self.on_step(step)
            steps = steps + [step]
            new_messages.append({"role": "tool", "content": text, "tool_name": name})
        return {"messages": state["messages"] + new_messages, "steps": steps}

    async def finalize(self, state: AgentState) -> dict:
        messages = state["messages"] + [{"role": "user", "content": FINALIZE_PROMPT}]
        _log("[llm] финальный ответ")
        response = await self._chat(messages, use_tools=False)
        step = {"type": "finalize", "reason": "лимит шагов" if state["iterations"] >= MAX_STEPS else "пустой ответ"}
        return {"answer": _strip_think(response.message.content), "steps": self._step(state, step)}

    def route(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools" if state["iterations"] < MAX_STEPS else "finalize"
        return END if state["answer"] else "finalize"

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve", self.retrieve)
        g.add_node("build_prompt", self.build_prompt)
        g.add_node("llm", self.call_llm)
        g.add_node("tools", self.call_tools)
        g.add_node("finalize", self.finalize)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "build_prompt")
        g.add_edge("build_prompt", "llm")
        g.add_conditional_edges("llm", self.route, {"tools": "tools", "finalize": "finalize", END: END})
        g.add_edge("tools", "llm")
        g.add_edge("finalize", END)
        return g.compile()

    async def run(self, question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        state: AgentState = {"question": question, "history": history or [], "context": [],
                             "messages": [], "steps": [], "iterations": 0, "answer": ""}
        result = await self.graph.ainvoke(state, config={"recursion_limit": 3 * MAX_STEPS + 10})
        return result["answer"], result["steps"]


async def run_agent(question: str, history: list[dict] | None = None,
                    on_step: StepCallback | None = None, model: str | None = None) -> tuple[str, list[dict]]:
    """Запускает MCP-сервер как подпроцесс, подключается к нему и выполняет запрос."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "src.mcp_server"], cwd=str(ROOT), env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = [mcp_tool_to_ollama(t) for t in listed.tools]
            if on_step:
                on_step({"type": "mcp", "tools": [t["function"]["name"] for t in tools]})
            agent = PCBuildAgent(session, tools, on_step, model)
            return await agent.run(question, history)
