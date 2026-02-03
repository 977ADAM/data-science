# src/config.py

from __future__ import annotations

import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

_raw_candidates = [
    BASE_DIR / "data" / "raw" / "Churn.csv",
    BASE_DIR / "Churn.csv",
]
RAW_DATA = next((p for p in _raw_candidates if p.exists()), _raw_candidates[0])

PROCESSED_DATA = BASE_DIR / "data" / "processed" / "clean.csv"

ARTIFACTS_DIR = BASE_DIR / "models"

# В проде выбираем версию модели через env:
# - MODEL_VERSION=20260202_120000Z_ab12cd
# - MODEL_VERSION=latest (по умолчанию)
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest").strip()

# Legacy path (backward compatibility)
LEGACY_MODEL_PATH = ARTIFACTS_DIR / "churn_pipeline.pkl"

# Optional uplift model artifact name (two-model T-learner)
UPLIFT_MODEL_FILENAME = "uplift_tlearner.pkl"

UPLIFT_MODEL_VERSION = os.getenv("UPLIFT_MODEL_VERSION", "latest").strip()

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

def _validate_model_version(v: str) -> str:
    """
    Защита от path traversal / неожиданных путей.
    Разрешаем только безопасный "slug" (буквы/цифры/._-), длина <= 64.
    """
    v = (v or "").strip()
    if v in ("", "latest"):
        return "latest"
    if not _VERSION_RE.match(v):
        raise ValueError(f"Invalid MODEL_VERSION: {v!r}")
    return v

def resolve_model_dir(version: str | None = None) -> Path:
    """
    Определяет директорию артефакта модели.
    Приоритет:
      1) явный version
      2) env MODEL_VERSION
      3) models/latest (symlink/dir)
      4) models/latest.txt (текстовый указатель)
    """
    v = _validate_model_version(version or MODEL_VERSION or "latest")
    if v and v != "latest":
        return ARTIFACTS_DIR / v

    # symlink/dir latest
    latest_dir = ARTIFACTS_DIR / "latest"
    if latest_dir.exists():
        return latest_dir

    # textual pointer (portable)
    latest_txt = ARTIFACTS_DIR / "latest.txt"
    if latest_txt.exists():
        try:
            target = latest_txt.read_text(encoding="utf-8").strip()
            if target:
                p = ARTIFACTS_DIR / target
                if p.exists():
                    return p
        except Exception:
            pass

    # fallback
    return ARTIFACTS_DIR / "latest"

def resolve_model_path(version: str | None = None) -> Path:
    """
    Возвращает путь до pipeline.pkl.
    Если не найдено — падаем назад на legacy churn_pipeline.pkl.
    """
    d = resolve_model_dir(version)
    candidate = d / "churn_pipeline.pkl"
    if candidate.exists():
        return candidate
    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    return candidate



def resolve_uplift_model_path(version: str | None = None) -> Path:
    """
    Возвращает путь до uplift T-learner артефакта.
    По умолчанию ищет в отдельном uplift "latest" указателе.
    """
    d = resolve_uplift_model_dir(version)
    return d / UPLIFT_MODEL_FILENAME


def resolve_uplift_model_dir(version: str | None = None) -> Path:
    """
    Определяет директорию uplift-артефакта модели.
    Приоритет:
      1) явный version
      2) env UPLIFT_MODEL_VERSION
      3) models/latest_uplift (symlink/dir)
      4) models/latest_uplift.txt (текстовый указатель)
    """
    v = _validate_model_version(version or UPLIFT_MODEL_VERSION or "latest")
    if v and v != "latest":
        return ARTIFACTS_DIR / v

    # symlink/dir latest_uplift
    latest_dir = ARTIFACTS_DIR / "latest_uplift"
    if latest_dir.exists():
        return latest_dir

    # textual pointer (portable)
    latest_txt = ARTIFACTS_DIR / "latest_uplift.txt"
    if latest_txt.exists():
        try:
            target = latest_txt.read_text(encoding="utf-8").strip()
            if target:
                p = ARTIFACTS_DIR / target
                if p.exists():
                    return p
        except Exception:
            pass

    # fallback
    return ARTIFACTS_DIR / "latest_uplift"

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5
