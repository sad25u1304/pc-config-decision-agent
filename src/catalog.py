"""Каталог комплектующих в SQLite: точные характеристики и цены.

Векторная БД отвечает на вопрос «что подходит по смыслу», а эта база —
«что подходит точно» (сокет, тип памяти, мощность, размеры, цена).
"""
import csv
import sqlite3
from pathlib import Path

from .config import CATALOG_CSV, DB_PATH

INT_FIELDS = {
    "price", "tdp", "igpu", "box_cooler", "max_gpu_len", "max_cooler_h",
    "length_mm", "height_mm", "watt", "capacity_gb",
}
COLUMNS = [
    "id", "category", "name", "price", "socket", "mem_type", "tdp", "igpu", "box_cooler",
    "form_factor", "supported_ff", "max_gpu_len", "max_cooler_h", "length_mm", "height_mm",
    "watt", "capacity_gb", "description",
]


def _convert(row: dict) -> dict:
    out = {}
    for col in COLUMNS:
        value = (row.get(col) or "").strip()
        if value == "":
            out[col] = None
        elif col in INT_FIELDS:
            out[col] = int(value)
        else:
            out[col] = value
    return out


def load_csv(csv_path: Path = CATALOG_CSV) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        return [_convert(r) for r in csv.DictReader(f)]


def build_db(csv_path: Path = CATALOG_CSV, db_path: Path = DB_PATH) -> int:
    """Пересоздаёт таблицу components из CSV. Возвращает число записей."""
    rows = load_csv(csv_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    col_defs = ", ".join(
        f"{c} {'INTEGER' if c in INT_FIELDS else 'TEXT'}" + (" PRIMARY KEY" if c == "id" else "")
        for c in COLUMNS
    )
    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE IF EXISTS components")
        con.execute(f"CREATE TABLE components ({col_defs})")
        placeholders = ", ".join("?" for _ in COLUMNS)
        con.executemany(
            f"INSERT INTO components ({', '.join(COLUMNS)}) VALUES ({placeholders})",
            [[r[c] for c in COLUMNS] for r in rows],
        )
    return len(rows)


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"База {db_path} не найдена. Сначала выполните: python -m src.build_index"
        )
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def get_components(ids: list[str], db_path: Path = DB_PATH) -> tuple[list[dict], list[str]]:
    """Возвращает (найденные комплектующие в порядке ids, список неизвестных id)."""
    ids = [i.strip() for i in ids if i and i.strip()]
    if not ids:
        return [], []
    with _connect(db_path) as con:
        q = f"SELECT * FROM components WHERE id IN ({', '.join('?' for _ in ids)})"
        found = {r["id"]: dict(r) for r in con.execute(q, ids)}
    return [found[i] for i in ids if i in found], [i for i in ids if i not in found]


def search(
    category: str | None = None,
    max_price: int | None = None,
    socket: str | None = None,
    mem_type: str | None = None,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """Точный поиск по фильтрам. Для кулеров socket ищется в списке поддерживаемых сокетов."""
    where, params = [], []
    if category:
        where.append("category = ?")
        params.append(category)
    if max_price:
        where.append("price <= ?")
        params.append(int(max_price))
    if socket:
        where.append("(socket = ? OR socket LIKE ?)")
        params += [socket, f"%{socket}%"]
    if mem_type:
        where.append("mem_type = ?")
        params.append(mem_type.upper())
    sql = "SELECT * FROM components"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY price"
    with _connect(db_path) as con:
        return [dict(r) for r in con.execute(sql, params)]
