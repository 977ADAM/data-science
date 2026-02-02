# src/versioning.py

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: object) -> str:
    # детерминированная сериализация
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(s)


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, p)


def atomic_write_json(path: str | Path, obj: object) -> None:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)
    atomic_write_text(path, text + "\n")


def package_versions(names: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:
            # пакет может отсутствовать (или имя отличается) — это нормально
            continue
    return out


def code_fingerprint(root: Path, include_globs: Iterable[str]) -> dict[str, str]:
    """
    Хэшируем содержимое файлов кода. Возвращаем:
      - per_file_sha256: {relpath: sha}
      - code_sha256: sha(json(per_file_sha256))
    """
    per: dict[str, str] = {}
    for g in include_globs:
        for p in sorted(root.glob(g)):
            if p.is_file():
                rel = str(p.relative_to(root)).replace("\\", "/")
                per[rel] = sha256_file(p)
    return {"per_file_sha256": per, "code_sha256": sha256_json(per)}


@dataclass(frozen=True)
class Manifest:
    model_version: str
    created_at_utc: str
    data: dict
    code: dict
    environment: dict
    training: dict
    artifacts: dict

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "created_at_utc": self.created_at_utc,
            "data": self.data,
            "code": self.code,
            "environment": self.environment,
            "training": self.training,
            "artifacts": self.artifacts,
        }


def build_manifest(
    *,
    model_version: str,
    repo_root: Path,
    raw_data_path: Path | None,
    training_params: dict,
    artifacts: dict,
    extra_code_globs: Iterable[str] = ("src/**/*.py", "app/**/*.py", "pyproject.toml", "requirements*.txt", "Dockerfile"),
) -> Manifest:
    created = datetime.now(timezone.utc).isoformat()

    data_block: dict = {}
    if raw_data_path is not None and raw_data_path.exists():
        data_block = {
            "raw_data_path": str(raw_data_path),
            "raw_data_sha256": sha256_file(raw_data_path),
        }
    else:
        data_block = {"raw_data_path": str(raw_data_path) if raw_data_path else None}

    code_block = code_fingerprint(repo_root, extra_code_globs)

    env_block = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(["numpy", "pandas", "scikit-learn", "catboost", "fastapi", "pydantic", "joblib", "uvicorn"]),
    }

    return Manifest(
        model_version=model_version,
        created_at_utc=created,
        data=data_block,
        code=code_block,
        environment=env_block,
        training=training_params,
        artifacts=artifacts,
    )
