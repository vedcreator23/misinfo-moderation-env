"""
Inference Script — MisinformationModerationEnv
===============================================
MANDATORY STDOUT FORMAT:
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL   — LLM inference endpoint (default: HF router)
  MODEL_NAME     — Model identifier to use for inference
  HF_TOKEN       — HuggingFace / API key (NO DEFAULT)
  LOCAL_IMAGE_NAME — Optional for docker images
  BASE_URL       — MisinformationModerationEnv URL (default: localhost:7860)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import List, Optional

import httpx
from openai import OpenAI

# ─────────────────────────── Configuration ──────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

# For the environment server URL
ENV_BASE_URL: str = os.getenv("BASE_URL", "http://localhost:7860")
BENCHMARK: str    = "misinfo-moderation-env"
MAX_STEPS: int    = 25
SUCCESS_SCORE_THRESHOLD: float = 0.5

# Max theoretical reward per step for normalisation fallback
MAX_STEP_REWARD = (1.0 * 1.3 + 0.1 + 1.0) / 2.5

# ─────────────────────────── Log Helpers ────────────────────────────

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_str = error if error else "null"
    done_str  = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    success_str  = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────── Env Client ─────────────────────────────

_http: Optional[httpx.AsyncClient] = None

async def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=30.0)
    return _http

async def env_reset(task: str) -> dict:
    c = await get_http()
    r = await c.post("/reset", json={"task": task})
    r.raise_for_status()
    return r.json()

async def env_step(action_type: str, post_id: Optional[str], user_id: Optional[str]) -> dict:
    c = await get_http()
    payload = {"action_type": action_type, "post_id": post_id, "user_id": user_id}
    r = await c.post("/step", json=payload)
    r.raise_for_status()
    return r.json()

async def env_close() -> None:
    global _http
    if _http:
        await _http.aclose()
        _http = None

# ─────────────────────────── Agent Logic ────────────────────────────

SYSTEM_PROMPT = """You are a content moderation AI agent.
Available actions: [flag_content, fact_check, reduce_visibility, ban_user, ignore]
Respond with JSON: {"action_type": "<action>", "reasoning": "<reason>"}"""

def get_agent_action(client: OpenAI, obs: dict) -> dict:
    post = obs["current_post"]
    user = obs["current_user"]
    prompt = f"Post: {post['content']}\nTruth: {post['truth_score']}\nAuthor Credibility: {user['credibility_score']}"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw: raw = raw.split("```")[1].replace("json", "")
        return json.loads(raw)
    except Exception:
        return {"action_type": "ignore"}

# ─────────────────────────── Task Runner ────────────────────────────

async def run_task(task_name: str, client: OpenAI) -> float:
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env_reset(task_name)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done: break

            action = get_agent_action(client, obs)
            action_str = action.get("action_type", "ignore")
            error = None

            try:
                result = await env_step(
                    action_type=action_str,
                    post_id=obs["current_post"]["post_id"] if action_str != "ban_user" else None,
                    user_id=obs["current_user"]["user_id"] if action_str == "ban_user" else None,
                )
                reward = result["reward"]
                done = result["done"]
                obs = result["observation"]
                if done: score = result["info"].get("final_score", 0.0)
            except Exception as e:
                reward = 0.0; done = True; error = str(e)[:50]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        score = round(min(1.0, max(0.0, score)), 4)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score

# ─────────────────────────── Main ───────────────────────────────────

async def main() -> None:
    if not HF_TOKEN:
        print("Error: HF_TOKEN is required.", file=sys.stderr)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    for task in ["easy", "medium", "hard"]:
        await run_task(task, client)
        await asyncio.sleep(1) # Breath
    
    await env_close()

if __name__ == "__main__":
    asyncio.run(main())
