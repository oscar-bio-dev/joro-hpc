#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/joro_db.py
Parámetros de conexión a PostGIS leídos del entorno (sin credenciales en el código).

Variables: JORO_DB_HOST (localhost), JORO_DB_PORT (5433), JORO_DB_NAME (joro_db),
JORO_DB_USER (joro_admin), JORO_DB_PASSWORD (obligatoria al conectar). Si existe un archivo `.env`
en la raíz del repositorio, se cargan sus pares CLAVE=VALOR sin sobrescribir el entorno actual.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

_ENV_LOADED = False


def _load_dotenv() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get_db_params() -> dict:
    _load_dotenv()
    return {
        "host": os.environ.get("JORO_DB_HOST") or "localhost",
        "port": int(os.environ.get("JORO_DB_PORT") or 5433),
        "dbname": os.environ.get("JORO_DB_NAME") or "joro_db",
        "user": os.environ.get("JORO_DB_USER") or "joro_admin",
        "password": os.environ.get("JORO_DB_PASSWORD") or "",
    }


def get_db_url(require_password: bool = True) -> str:
    """URL libpq `postgresql://user:pass@host:port/db` construida desde el entorno."""
    p = get_db_params()
    if require_password and not p["password"]:
        raise RuntimeError(
            "JORO_DB_PASSWORD no está definida. Copie .env.example a .env y complete la contraseña, "
            "o exporte la variable de entorno antes de ejecutar este script."
        )
    cred = quote(p["user"], safe="")
    if p["password"]:
        cred += ":" + quote(p["password"], safe="")
    return f"postgresql://{cred}@{p['host']}:{p['port']}/{p['dbname']}"
