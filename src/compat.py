"""Детерминированная проверка совместимости сборки.

LLM плохо сравнивает числа и может «уговорить себя», что детали подходят,
поэтому все проверки сделаны обычным кодом, а LLM получает готовый вердикт.
"""
from collections import defaultdict

from .config import CATEGORY_RU

REQUIRED = ["cpu", "motherboard", "ram", "storage", "psu", "case"]
SINGLE = ["cpu", "motherboard", "psu", "case", "cooler"]
BASE_POWER_W = 80          # плата, память, накопители, вентиляторы
PSU_MIN_FACTOR = 1.2       # меньше — ошибка
PSU_GOOD_FACTOR = 1.4      # меньше — предупреждение


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(";") if v.strip()]


def estimate_power(parts: list[dict]) -> int:
    cpu = sum(p["tdp"] or 0 for p in parts if p["category"] == "cpu")
    gpu = sum(p["tdp"] or 0 for p in parts if p["category"] == "gpu")
    return cpu + gpu + BASE_POWER_W


def check_build(parts: list[dict]) -> dict:
    by = defaultdict(list)
    for p in parts:
        by[p["category"]].append(p)
    errors, warnings = [], []

    for cat in SINGLE:
        if len(by[cat]) > 1:
            errors.append(f"В сборке несколько позиций категории «{CATEGORY_RU[cat]}», нужна одна.")
    if len(by["gpu"]) > 1:
        warnings.append("В сборке несколько видеокарт — обычно нужна одна.")

    for cat in REQUIRED:
        if not by[cat]:
            errors.append(f"Не хватает комплектующего: {CATEGORY_RU[cat]}.")

    cpu = by["cpu"][0] if by["cpu"] else None
    mb = by["motherboard"][0] if by["motherboard"] else None
    case = by["case"][0] if by["case"] else None
    cooler = by["cooler"][0] if by["cooler"] else None
    psu = by["psu"][0] if by["psu"] else None
    gpus = by["gpu"]

    # 1. Сокет процессора и платы
    if cpu and mb and cpu["socket"] != mb["socket"]:
        errors.append(
            f"Сокет процессора {cpu['name']} ({cpu['socket']}) не совпадает "
            f"с сокетом платы {mb['name']} ({mb['socket']})."
        )

    # 2. Тип памяти
    if mb:
        for ram in by["ram"]:
            if ram["mem_type"] != mb["mem_type"]:
                errors.append(
                    f"Память {ram['name']} ({ram['mem_type']}) не подходит к плате "
                    f"{mb['name']}, которая поддерживает только {mb['mem_type']}."
                )

    # 3. Графика
    if cpu and not gpus and not cpu["igpu"]:
        errors.append(
            f"У процессора {cpu['name']} нет встроенной графики — нужна видеокарта."
        )

    # 4. Кулер
    if cpu and not cooler and not cpu["box_cooler"]:
        errors.append(f"Процессор {cpu['name']} продаётся без кулера — добавьте кулер.")
    if cooler and cpu:
        if cpu["socket"] not in _split(cooler["socket"]):
            errors.append(f"Кулер {cooler['name']} не поддерживает сокет {cpu['socket']}.")
        if (cooler["tdp"] or 0) < (cpu["tdp"] or 0):
            errors.append(
                f"Кулер {cooler['name']} рассчитан на {cooler['tdp']} Вт, "
                f"а процессор {cpu['name']} выделяет {cpu['tdp']} Вт."
            )

    # 5. Корпус
    if case and mb and mb["form_factor"] not in _split(case["supported_ff"]):
        errors.append(
            f"Плата формата {mb['form_factor']} не помещается в корпус {case['name']} "
            f"(поддерживает: {case['supported_ff'].replace(';', ', ')})."
        )
    if case:
        for gpu in gpus:
            if (gpu["length_mm"] or 0) > (case["max_gpu_len"] or 0):
                errors.append(
                    f"Видеокарта {gpu['name']} длиной {gpu['length_mm']} мм не помещается "
                    f"в корпус {case['name']} (максимум {case['max_gpu_len']} мм)."
                )
        if cooler and (cooler["height_mm"] or 0) > (case["max_cooler_h"] or 0):
            errors.append(
                f"Кулер {cooler['name']} высотой {cooler['height_mm']} мм не помещается "
                f"в корпус {case['name']} (максимум {case['max_cooler_h']} мм)."
            )

    # 6. Блок питания
    power = estimate_power(parts)
    recommended = int(round(power * PSU_GOOD_FACTOR / 50.0) * 50)
    if psu:
        if psu["watt"] < power * PSU_MIN_FACTOR:
            errors.append(
                f"Блока питания {psu['watt']} Вт недостаточно: расчётное потребление "
                f"{power} Вт, рекомендуется от {recommended} Вт."
            )
        elif psu["watt"] < power * PSU_GOOD_FACTOR:
            warnings.append(
                f"Блок питания {psu['watt']} Вт работает почти без запаса "
                f"(потребление {power} Вт, рекомендуется {recommended} Вт)."
            )

    # 7. Мягкие предупреждения
    if any((s["form_factor"] or "").startswith("3.5") for s in by["storage"]) and not any(
        (s["form_factor"] or "").startswith("M.2") for s in by["storage"]
    ):
        warnings.append("Система будет на жёстком диске — рекомендуется SSD NVMe.")
    for ram in by["ram"]:
        if "1×" in ram["name"]:
            warnings.append(f"{ram['name']} — один модуль, память будет работать в одноканальном режиме.")

    return {
        "compatible": not errors,
        "errors": errors,
        "warnings": warnings,
        "estimated_power_w": power,
        "recommended_psu_w": recommended,
    }
