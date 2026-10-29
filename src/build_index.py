"""Модуль 1. Индексация текстов предметной области.

Запуск:  python -m src.build_index

1. Каталог data/components.csv  ->  SQLite (storage/catalog.db) — точные характеристики.
2. Гайды data/knowledge/*.md    ->  чанки -> эмбеддинги bge-m3 -> Chroma, коллекция «knowledge».
3. Карточки комплектующих        ->  эмбеддинги -> Chroma, коллекция «components»
   (чтобы искать детали по смыслу: «тихий», «для видеосвязи», «компактный»).
"""
import time

import chromadb

from . import catalog
from .config import (
    CATEGORY_RU, CHROMA_DIR, COMPONENTS_COLLECTION, EMBED_MODEL, KNOWLEDGE_COLLECTION,
    KNOWLEDGE_DIR,
)
from .rag import chunk_text, embed


def component_card(c: dict) -> str:
    """Текстовая карточка комплектующего для векторной БД."""
    specs = []
    for key, label, unit in [
        ("socket", "сокет", ""), ("mem_type", "память", ""), ("tdp", "TDP", " Вт"),
        ("form_factor", "форм-фактор", ""), ("capacity_gb", "объём", " ГБ"),
        ("watt", "мощность", " Вт"), ("length_mm", "длина", " мм"), ("height_mm", "высота", " мм"),
    ]:
        if c.get(key):
            specs.append(f"{label} {c[key]}{unit}")
    if c["category"] == "cpu":
        specs.append("есть встроенная графика" if c["igpu"] else "нет встроенной графики")
    return (
        f"{CATEGORY_RU[c['category']]} {c['name']}. {c['description']}. "
        f"Характеристики: {', '.join(specs)}. Цена {c['price']} ₽."
    )


def add_in_batches(collection, ids, docs, metas, batch: int = 32) -> None:
    for i in range(0, len(ids), batch):
        part = docs[i:i + batch]
        collection.add(
            ids=ids[i:i + batch], documents=part, metadatas=metas[i:i + batch],
            embeddings=embed(part),
        )
        print(f"  проиндексировано {min(i + batch, len(ids))}/{len(ids)}")


def main() -> None:
    t0 = time.time()
    n = catalog.build_db()
    print(f"[1/3] SQLite: загружено {n} комплектующих")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    for name in (KNOWLEDGE_COLLECTION, COMPONENTS_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    # --- гайды
    kn = client.create_collection(KNOWLEDGE_COLLECTION, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    for f in files:
        text = f.read_text(encoding="utf-8")
        title, _, body = text.partition("\n")
        title = title.lstrip("# ").strip()
        for i, chunk in enumerate(chunk_text(body)):
            ids.append(f"{f.stem}#{i}")
            docs.append(f"{title}\n{chunk}")
            metas.append({"source": f.name, "title": title})
    print(f"[2/3] База знаний: {len(files)} документов -> {len(ids)} чанков, модель {EMBED_MODEL}")
    add_in_batches(kn, ids, docs, metas)

    # --- карточки комплектующих
    comp = client.create_collection(COMPONENTS_COLLECTION, metadata={"hnsw:space": "cosine"})
    rows = catalog.load_csv()
    print(f"[3/3] Карточки комплектующих: {len(rows)}")
    add_in_batches(
        comp,
        [r["id"] for r in rows],
        [component_card(r) for r in rows],
        [{"source": f"catalog:{r['id']}", "category": r["category"], "name": r["name"]} for r in rows],
    )
    print(f"Готово за {time.time() - t0:.1f} c. Индекс: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
