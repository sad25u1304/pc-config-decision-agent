"""Модуль 3. MCP-сервер с инструментами агента (FastMCP, транспорт stdio).

Запускается агентом автоматически. Для отладки можно открыть MCP Inspector:
    mcp dev src/mcp_server.py
Важно: сервер общается через stdout, поэтому весь лог пишется в stderr.
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import json
import logging
import re
import sys
from datetime import datetime
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from . import catalog, compat
from .config import (
    BUILDS_DIR, CATEGORIES, CATEGORY_RU, COMPONENTS_COLLECTION, KNOWLEDGE_COLLECTION,
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[mcp] %(message)s")
log = logging.getLogger("pc-builder")

mcp = FastMCP("pc-builder")
_retrievers: dict = {}


def _retriever(name: str):
    """Ленивая загрузка гибридного поиска; при ошибке возвращает None (работаем без него)."""
    if name not in _retrievers:
        try:
            from .rag import HybridRetriever
            _retrievers[name] = HybridRetriever(name)
        except Exception as e:  # нет индекса или Ollama недоступна
            log.warning("retriever %s недоступен: %s", name, e)
            _retrievers[name] = None
    return _retrievers[name]


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _short(c: dict) -> dict:
    """Краткое описание детали: только поля, важные для её категории."""
    keys = {
        "cpu": ["socket", "tdp", "igpu", "box_cooler"],
        "motherboard": ["socket", "mem_type", "form_factor"],
        "ram": ["mem_type", "capacity_gb"],
        "gpu": ["tdp", "length_mm"],
        "storage": ["form_factor", "capacity_gb"],
        "psu": ["watt"],
        "case": ["supported_ff", "max_gpu_len", "max_cooler_h"],
        "cooler": ["socket", "tdp", "height_mm"],
    }[c["category"]]
    out = {"id": c["id"], "name": c["name"], "price": c["price"]}
    out.update({k: c[k] for k in keys if c.get(k) is not None})
    out["description"] = c["description"]
    return out


@mcp.tool()
def search_knowledge(
    query: Annotated[str, Field(description="Вопрос или тема для поиска в базе знаний")],
    k: Annotated[int, Field(description="Сколько фрагментов вернуть")] = 3,
) -> str:
    """Ищет в базе знаний (гайды по выбору комплектующих) фрагменты по запросу.
    Используй, если нужно уточнить требования к задаче или правило выбора детали.
    Возвращает фрагменты текста и имя файла-источника для ссылки."""
    r = _retriever(KNOWLEDGE_COLLECTION)
    if r is None:
        return _dump({"error": "база знаний недоступна"})
    hits = r.search(query, k=k)
    return _dump([{"source": h["source"], "text": h["text"]} for h in hits])


@mcp.tool()
def search_components(
    query: Annotated[str, Field(description="Что нужно, своими словами: «тихий кулер», «видеокарта для монтажа»")],
    category: Annotated[str, Field(description="Категория: cpu, motherboard, ram, gpu, storage, psu, case, cooler")],
    max_price: Annotated[int | None, Field(description="Максимальная цена в рублях")] = None,
    socket: Annotated[str | None, Field(description="Сокет: AM5, AM4 или LGA1700 (для cpu, motherboard, cooler)")] = None,
    mem_type: Annotated[str | None, Field(description="Тип памяти: DDR4 или DDR5 (для motherboard, ram)")] = None,
    limit: Annotated[int, Field(description="Сколько вариантов вернуть")] = 4,
) -> str:
    """Ищет комплектующие в каталоге магазина: по смыслу запроса (векторная БД)
    и по точным фильтрам (SQLite). Возвращает id, название, цену и характеристики.
    Используй в сборке только id, полученные этим инструментом."""
    category = category.strip().lower()
    if category not in CATEGORIES:
        return _dump({"error": f"неизвестная категория {category}", "allowed": CATEGORIES})

    candidates = catalog.search(category=category, max_price=max_price, socket=socket, mem_type=mem_type)
    if not candidates:
        return _dump({"found": 0, "hint": "ничего не найдено — ослабь фильтры или увеличь max_price"})

    # Ранжирование по смыслу: порядок из гибридного поиска по карточкам
    r = _retriever(COMPONENTS_COLLECTION)
    if r is not None:
        ranked = [h["id"] for h in r.search(query, k=40, where={"category": category}, pool=40)]
        pos = {cid: i for i, cid in enumerate(ranked)}
        candidates.sort(key=lambda c: (pos.get(c["id"], 999), c["price"]))
    return _dump({"found": len(candidates), "items": [_short(c) for c in candidates[:limit]]})


@mcp.tool()
def check_compatibility(
    component_ids: Annotated[list[str], Field(description="Список id всех деталей сборки, например [\"cpu-02\", \"mb-02\"]")],
) -> str:
    """Проверяет совместимость сборки: сокет, тип памяти, форм-фактор, размеры
    видеокарты и кулера, мощность блока питания, наличие графики и кулера.
    Если compatible=false — исправь детали из errors и проверь снова."""
    parts, unknown = catalog.get_components(component_ids)
    result = compat.check_build(parts)
    if unknown:
        result["compatible"] = False
        result["errors"].insert(0, f"Неизвестные id (их нет в каталоге): {', '.join(unknown)}")
    result["total_price"] = sum(p["price"] for p in parts)
    return _dump(result)


@mcp.tool()
def calc_total(
    component_ids: Annotated[list[str], Field(description="Список id всех деталей сборки")],
    budget: Annotated[int, Field(description="Бюджет пользователя в рублях")],
) -> str:
    """Считает итоговую стоимость сборки и сравнивает её с бюджетом."""
    parts, unknown = catalog.get_components(component_ids)
    total = sum(p["price"] for p in parts)
    return _dump({
        "items": [{"id": p["id"], "name": p["name"], "price": p["price"]} for p in parts],
        "unknown_ids": unknown,
        "total": total,
        "budget": budget,
        "remaining": budget - total,
        "within_budget": total <= budget,
    })


@mcp.tool()
def save_build(
    title: Annotated[str, Field(description="Короткое название сборки")],
    component_ids: Annotated[list[str], Field(description="Список id всех деталей итоговой сборки")],
    comment: Annotated[str, Field(description="Короткое обоснование выбора")] = "",
) -> str:
    """Сохраняет итоговую сборку в файл Markdown (спецификация для покупки).
    Вызывай только после успешной проверки совместимости и бюджета."""
    parts, unknown = catalog.get_components(component_ids)
    if unknown or not parts:
        return _dump({"error": f"неизвестные id: {unknown}"})
    check = compat.check_build(parts)
    total = sum(p["price"] for p in parts)

    lines = [f"# {title}", "", f"Дата: {datetime.now():%d.%m.%Y %H:%M}", "",
             "| Категория | Модель | Цена, ₽ |", "|---|---|---:|"]
    lines += [f"| {CATEGORY_RU[p['category']]} | {p['name']} | {p['price']:,} |".replace(",", " ") for p in parts]
    lines += [f"| **Итого** | | **{total:,}** |".replace(",", " "), ""]
    lines.append(f"Совместимость: {'✅ проверена' if check['compatible'] else '❌ есть ошибки'}; "
                 f"расчётное потребление {check['estimated_power_w']} Вт.")
    for w in check["warnings"]:
        lines.append(f"- ⚠️ {w}")
    if comment:
        lines += ["", "## Обоснование", "", comment]

    BUILDS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^\w-]+", "_", title.lower()).strip("_")[:40] or "build"
    path = BUILDS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("сборка сохранена: %s", path)
    return _dump({"saved": True, "path": str(path), "total": total, "compatible": check["compatible"]})


if __name__ == "__main__":
    mcp.run()  # транспорт stdio
