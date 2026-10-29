"""Настройки проекта. Любой параметр можно переопределить переменной окружения."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Данные предметной области
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"      # гайды (.md) -> векторная БД
CATALOG_CSV = DATA_DIR / "components.csv"   # каталог комплектующих -> SQLite + векторная БД

# Хранилища, которые создаёт build_index.py
STORAGE_DIR = ROOT / "storage"
DB_PATH = STORAGE_DIR / "catalog.db"
CHROMA_DIR = STORAGE_DIR / "chroma"
BUILDS_DIR = ROOT / "builds"                # сюда инструмент save_build сохраняет сборки

# Модели (Ollama)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")     # для 8 ГБ ОЗУ: qwen3:4b
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))        # размер контекста LLM
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Параметры RAG и агента
CHUNK_SIZE = 700          # символов в чанке
CHUNK_OVERLAP = 150       # перекрытие соседних чанков
TOP_K = 4                 # сколько фрагментов подкладывать в промпт
MAX_STEPS = int(os.getenv("MAX_STEPS", "14"))   # предел шагов цикла «LLM -> инструменты»

KNOWLEDGE_COLLECTION = "knowledge"
COMPONENTS_COLLECTION = "components"

CATEGORIES = ["cpu", "motherboard", "ram", "gpu", "storage", "psu", "case", "cooler"]
CATEGORY_RU = {
    "cpu": "Процессор",
    "motherboard": "Материнская плата",
    "ram": "Оперативная память",
    "gpu": "Видеокарта",
    "storage": "Накопитель",
    "psu": "Блок питания",
    "case": "Корпус",
    "cooler": "Кулер",
}
