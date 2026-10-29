"""RAG-часть: нарезка текстов, эмбеддинги и гибридный поиск (векторный + BM25)."""
import re
from collections import defaultdict

from .config import (
    CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, OLLAMA_HOST, TOP_K,
)

_TOKEN = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


# ---------------------------------------------------------------- чанкинг
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Режет текст по абзацам в куски до size символов.

    Если абзац не помещается целиком, он режется скользящим окном.
    Последний короткий абзац предыдущего чанка повторяется в начале следующего
    (перекрытие), чтобы мысль не обрывалась на границе.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= size:
            pieces.append(p)
        else:
            step = size - overlap
            pieces += [p[i:i + size] for i in range(0, len(p), step)]

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            last = current.split("\n\n")[-1]
            current = (last + "\n\n" if len(last) <= overlap else "") + piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------- эмбеддинги
_client = None


def embed(texts: list[str], batch: int = 16) -> list[list[float]]:
    """Превращает тексты в векторы моделью bge-m3 через Ollama."""
    global _client
    import ollama
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    vectors = []
    for i in range(0, len(texts), batch):
        res = _client.embed(model=EMBED_MODEL, input=texts[i:i + batch])
        vectors += res["embeddings"]
    return vectors


# ---------------------------------------------------------------- поиск
def _matches(meta: dict, where: dict | None) -> bool:
    return not where or all(meta.get(k) == v for k, v in where.items())


class HybridRetriever:
    """Гибридный поиск по коллекции Chroma.

    1) векторный поиск по смыслу (Chroma + bge-m3);
    2) лексический поиск BM25 — ловит точные названия вроде «8600G» или «DDR5»;
    3) результаты объединяются методом Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, collection_name: str):
        import chromadb
        from rank_bm25 import BM25Okapi

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(collection_name)
        data = self.collection.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.docs = data["documents"]
        self.metas = data["metadatas"]
        if not self.ids:
            raise RuntimeError(f"Коллекция {collection_name} пуста — запустите build_index")
        self.index = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])

    def search(self, query: str, k: int = TOP_K, where: dict | None = None, pool: int = 20) -> list[dict]:
        n = min(pool, len(self.ids))

        # 1. Векторный поиск
        vec = self.collection.query(
            query_embeddings=embed([query]), n_results=n, where=where or None,
        )
        vec_ids = vec["ids"][0] if vec["ids"] else []

        # 2. BM25
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)
        bm_ids = [self.ids[i] for i in order if scores[i] > 0 and _matches(self.metas[i], where)][:n]

        # 3. Reciprocal Rank Fusion
        fused: dict[str, float] = defaultdict(float)
        for rank, doc_id in enumerate(vec_ids):
            fused[doc_id] += 1.0 / (60 + rank)
        for rank, doc_id in enumerate(bm_ids):
            fused[doc_id] += 1.0 / (60 + rank)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            {
                "id": doc_id,
                "text": self.docs[self.index[doc_id]],
                "score": round(fused[doc_id], 4),
                **self.metas[self.index[doc_id]],
            }
            for doc_id in top
        ]
