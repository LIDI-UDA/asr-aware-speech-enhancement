# utils/text.py
import unicodedata


def normalize_text_for_wer(text: str) -> str:
    """
    Normalización ÚNICA para WER en TODO el proyecto.

    - lower()
    - quitar tildes
    - quitar TODA puntuación unicode (incluye ¿¡)
    - colapsar espacios

    Devuelve string limpio (puede ser "").
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # quitar tildes
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

    # quitar puntuación unicode (categoría "P*": Pc, Pd, Pe, Pf, Pi, Po, Ps)
    text = "".join(
        " " if unicodedata.category(c).startswith("P") else c
        for c in text
    )

    # normalizar espacios
    text = " ".join(text.split())
    return text
