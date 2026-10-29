"""Тесты проверки совместимости и каталога (не требуют Ollama).
Запуск:  python -m unittest discover tests
"""
import tempfile
import unittest
from pathlib import Path

from src import catalog, compat
from src.rag import chunk_text


class CompatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "test.db"
        cls.count = catalog.build_db(db_path=cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, ids):
        parts, unknown = catalog.get_components(ids, db_path=self.db)
        self.assertEqual(unknown, [])
        return compat.check_build(parts)

    def test_catalog_loaded(self):
        self.assertGreaterEqual(self.count, 60)

    def test_office_build_ok(self):
        r = self.check(["cpu-02", "mb-02", "ram-03", "ssd-02", "psu-07", "case-03", "cool-03"])
        self.assertTrue(r["compatible"], r["errors"])

    def test_wrong_memory_type(self):
        r = self.check(["cpu-02", "mb-02", "ram-01", "ssd-02", "psu-07", "case-03"])
        self.assertFalse(r["compatible"])
        self.assertTrue(any("DDR4" in e for e in r["errors"]))

    def test_wrong_socket(self):
        r = self.check(["cpu-10", "mb-02", "ram-03", "ssd-02", "psu-02", "case-01"])
        self.assertTrue(any("Сокет" in e for e in r["errors"]))

    def test_cpu_without_graphics_needs_gpu(self):
        r = self.check(["cpu-12", "mb-04", "ram-01", "ssd-02", "psu-02", "case-01"])
        self.assertTrue(any("встроенной графики" in e for e in r["errors"]))

    def test_cpu_without_box_cooler(self):
        r = self.check(["cpu-05", "mb-03", "ram-04", "ssd-04", "gpu-06", "psu-04", "case-02"])
        self.assertTrue(any("без кулера" in e for e in r["errors"]))

    def test_weak_psu_for_4090(self):
        r = self.check(["cpu-05", "mb-03", "ram-04", "ssd-05", "gpu-09", "psu-03", "case-02", "cool-06"])
        self.assertTrue(any("Блока питания" in e for e in r["errors"]))

    def test_gpu_and_cooler_do_not_fit_compact_case(self):
        r = self.check(["cpu-04", "mb-02", "ram-04", "ssd-04", "gpu-06", "psu-04", "case-06", "cool-04"])
        self.assertTrue(any("Видеокарта" in e for e in r["errors"]))
        self.assertTrue(any("Кулер" in e and "высотой" in e for e in r["errors"]))

    def test_atx_board_in_matx_case(self):
        r = self.check(["cpu-02", "mb-03", "ram-03", "ssd-02", "psu-02", "case-03"])
        self.assertTrue(any("форм" in e.lower() or "помещается" in e for e in r["errors"]))

    def test_gaming_build_ok(self):
        r = self.check(["cpu-03", "mb-02", "ram-04", "ssd-04", "gpu-03", "psu-02", "case-02"])
        self.assertTrue(r["compatible"], r["errors"])


class ChunkTests(unittest.TestCase):
    def test_chunks_respect_size(self):
        text = "\n\n".join(["Абзац " + "слово " * 60] * 6)
        chunks = chunk_text(text, size=500, overlap=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 500 + 100 + 2 for c in chunks))


if __name__ == "__main__":
    unittest.main()
