"""
MisinformationModerationEnv – FastAPI Server App
=================================================
Creates the HTTP server that OpenEnv clients interact with.
Uses openenv-core's create_fastapi_app when available for full spec compliance.
"""
from __future__ import annotations

import sys
import os

# Ensure parent package is importable when running from server/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import (
    ModerationAction,
    ModerationObservation,
    ModerationState,
    ModerationStepResult,
)
from server.environment import MisinformationModerationEnvironment

# ─────────────────────────── OpenEnv integration ─────────────────────

# Attempt to use openenv-core's app factory for full spec compliance.
# Falls back to a plain FastAPI app if openenv-core is not installed.
try:
    from openenv.core.env_server import create_fastapi_app as _openenv_create_app  # type: ignore
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

# ─────────────────────────── App Setup ───────────────────────────────

_env = MisinformationModerationEnvironment(seed=42)

if _OPENENV_AVAILABLE:
    # Let openenv-core wire reset/step/state automatically
    app = _openenv_create_app(_env, ModerationAction, ModerationObservation)
else:
    app = FastAPI(
        title="MisinformationModerationEnv",
        description="OpenEnv-compliant RL environment for AI-driven social media moderation.",
        version="1.0.0",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── Request Models ──────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"


# ─────────────────────────── Routes ─────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "MisinformationModerationEnv", "openenv_core": _OPENENV_AVAILABLE}


@app.post("/reset", response_model=ModerationObservation)
def reset(req: ResetRequest = None):
    """Reset the environment and return the initial observation."""
    task = (req.task if req else "easy")
    try:
        obs = _env.reset(task=task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=ModerationStepResult)
def step(action: ModerationAction):
    """Execute one moderation action."""
    try:
        result = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state", response_model=ModerationState)
def state():
    """Return current episode state / metadata."""
    try:
        return _env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    from tasks import TASK_REGISTRY
    return {
        name: {
            "description": cfg.description,
            "max_steps": cfg.max_steps,
            "difficulty": cfg.name.value,
        }
        for name, cfg in TASK_REGISTRY.items()
    }


# ─────────────────────────── Entry point ────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
