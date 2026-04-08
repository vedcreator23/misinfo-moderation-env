"""
MisinformationModerationEnv – Client
=====================================
HTTP client for MisinformationModerationEnv.
Inherits from openenv-core EnvClient when available; falls back to a plain httpx wrapper.
"""
from __future__ import annotations

import httpx
from typing import Optional

from models import (
    ModerationAction,
    ModerationObservation,
    ModerationState,
    ModerationStepResult,
)

# ── openenv-core base class (optional) ───────────────────────────────
try:
    from openenv.core.client import EnvClient as _OpenEnvClient  # type: ignore
    _BASE = _OpenEnvClient
    _OPENENV = True
except ImportError:
    _BASE = object
    _OPENENV = False


class MisinformationModerationEnv(_BASE):
    """
    Synchronous HTTP client for MisinformationModerationEnv.

    Usage
    -----
    with MisinformationModerationEnv(base_url="http://localhost:7860") as env:
        obs  = env.reset(task="easy")
        result = env.step(ModerationAction(action_type="ignore"))
        state  = env.state
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.Client] = None
        self._timeout = timeout
        if _OPENENV:
            try:
                super().__init__(base_url=base_url)
            except Exception:
                pass

    # ── Context manager ──────────────────────────────────

    def __enter__(self):
        self._client = httpx.Client(base_url=self.base_url, timeout=self._timeout)
        return self

    def __exit__(self, *args):
        if self._client:
            self._client.close()

    def close(self):
        if self._client:
            self._client.close()

    # ── OpenEnv API ──────────────────────────────────────

    def reset(self, task: str = "easy") -> ModerationObservation:
        resp = self._client.post("/reset", json={"task": task})
        resp.raise_for_status()
        return ModerationObservation(**resp.json())

    def step(self, action: ModerationAction) -> ModerationStepResult:
        resp = self._client.post(
            "/step",
            content=action.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return ModerationStepResult(**resp.json())

    @property
    def state(self) -> ModerationState:
        resp = self._client.get("/state")
        resp.raise_for_status()
        return ModerationState(**resp.json())

    def list_tasks(self) -> dict:
        resp = self._client.get("/tasks")
        resp.raise_for_status()
        return resp.json()
