"""ASGI entry point: ``uvicorn orion.api.asgi:app``.

Kept separate from :mod:`orion.api.app` so the module-level object exists for
the server to import without every import of that module paying for a database
connection. Tests call ``create_app(...)`` directly and get their own store;
``uvicorn`` imports this and gets the configured one.

Configuration comes from the environment:

``ORION_DATABASE_URL``
    SQLAlchemy URL. Defaults to a local SQLite file, so ``uvicorn
    orion.api.asgi:app`` works in a clean checkout with no database server.
``ORION_EXPERIMENTS_ROOT``
    Where the benchmark evidence lives. Defaults to ``./experiments``.
"""

from __future__ import annotations

import os
from pathlib import Path

from orion.api.app import create_app
from orion.api.app import default_database_url, default_experiments_root

# Resolved at import so a misconfigured deployment fails immediately and
# visibly, rather than on the first request that touches the store.
DATABASE_URL = default_database_url()
EXPERIMENTS_ROOT = Path(default_experiments_root())

app = create_app(DATABASE_URL, experiments_root=EXPERIMENTS_ROOT)

__all__ = ["app", "DATABASE_URL", "EXPERIMENTS_ROOT"]
