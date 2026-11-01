"""Transactional, parameter-identified caches for append-only trajectory features."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable


class FeatureCache:
    """Keep completed frame features; reject reuse with a different analysis identity."""

    def __init__(self, path: Path, analysis_key: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS frames (source TEXT, step INTEGER, digest TEXT, "
            "payload TEXT, PRIMARY KEY(source,step))"
        )
        row = self.connection.execute("SELECT value FROM meta WHERE key='analysis_key'").fetchone()
        if row is not None and row[0] != analysis_key:
            self.connection.close()
            raise ValueError("Cache analysis identity changed; use a separately versioned cache")
        self.connection.execute("INSERT OR IGNORE INTO meta VALUES ('analysis_key',?)", (analysis_key,))
        self.connection.commit()

    def steps(self, source: str) -> set[int]:
        return {row[0] for row in self.connection.execute(
            "SELECT step FROM frames WHERE source=?", (source,))}

    def put(self, source: str, step: int, digest: str, payload: dict) -> None:
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        previous = self.connection.execute(
            "SELECT digest,payload FROM frames WHERE source=? AND step=?", (source, step)
        ).fetchone()
        if previous is not None and previous != (digest, encoded):
            raise ValueError(f"Existing frame identity/content changed at {source}:{step}")
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO frames VALUES (?,?,?,?)",
                                    (source, step, digest, encoded))

    def load(self, source: str, steps: Iterable[int]) -> list[dict]:
        result = []
        for step in steps:
            row = self.connection.execute(
                "SELECT payload FROM frames WHERE source=? AND step=?", (source, step)
            ).fetchone()
            if row is None:
                raise ValueError(f"Missing required complete frame {source}:{step}")
            result.append(json.loads(row[0]))
        return result

    def close(self) -> None:
        self.connection.close()
