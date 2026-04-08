"""
MisinformationModerationEnv – Server-side Environment
======================================================
Implements the OpenEnv Environment base class.
"""
from __future__ import annotations

import random
import uuid
from typing import Dict, List, Optional, Tuple

from models import (
    ActionType,
    ModerationAction,
    ModerationObservation,
    ModerationState,
    ModerationStepResult,
    PlatformMetrics,
    PostInfo,
    TaskDifficulty,
    UserInfo,
)
from tasks import (
    TASK_REGISTRY,
    TaskConfig,
    compute_step_reward,
    generate_episode,
    grade_episode,
)


class MisinformationModerationEnvironment:
    """
    OpenEnv-compatible environment for social-media content moderation.

    Lifecycle
    ---------
    env = MisinformationModerationEnvironment()
    obs = env.reset(task="easy")
    result = env.step(action)   # ModerationStepResult
    s     = env.state           # ModerationState
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng = random.Random(seed)
        self._task_cfg: Optional[TaskConfig] = None
        self._episode: List[Tuple[Dict, Dict]] = []
        self._step_idx: int = 0
        self._episode_id: str = ""

        # Accumulators
        self._total_reward: float = 0.0
        self._correct: int = 0
        self._false_positives: int = 0
        self._false_negatives: int = 0
        self._unnecessary_bans: int = 0
        self._flagged_posts: int = 0
        self._spread_rate: float = 0.0

        self._last_result: Optional[str] = None
        self._last_reward: Optional[float] = None
        self._done: bool = False

    # ─────────────────── OpenEnv API ────────────────────

    def reset(self, task: str = "easy") -> ModerationObservation:
        """Initialise a new episode and return the first observation."""
        if task not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_REGISTRY)}")

        self._task_cfg = TASK_REGISTRY[task]
        self._rng = random.Random(self._seed)
        self._episode = generate_episode(self._task_cfg, self._rng)
        self._step_idx = 0
        self._episode_id = str(uuid.uuid4())

        self._total_reward = 0.0
        self._correct = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._unnecessary_bans = 0
        self._flagged_posts = 0
        self._spread_rate = 0.0
        self._done = False
        self._last_result = None
        self._last_reward = None

        return self._make_observation()

    def step(self, action: ModerationAction) -> ModerationStepResult:
        """Execute one moderation action and return the full step result."""
        if self._task_cfg is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        post_data, user_data = self._episode[self._step_idx]

        reward, result_str, is_fp, is_fn = compute_step_reward(
            action=action.action_type,
            post=post_data,
            user=user_data,
            task=self._task_cfg,
            spread_rate=self._spread_rate,
        )

        # Update accumulators
        self._total_reward += reward
        self._last_result = result_str
        self._last_reward = reward

        if is_fp:
            self._false_positives += 1
            if action.action_type == ActionType.BAN_USER:
                self._unnecessary_bans += 1
        elif is_fn:
            self._false_negatives += 1
        else:
            self._correct += 1

        # Update platform spread rate
        if action.action_type in (ActionType.FLAG_CONTENT, ActionType.FACT_CHECK, ActionType.REDUCE_VISIBILITY):
            self._flagged_posts += 1

        total_posts = self._step_idx + 1
        missed = (self._step_idx + 1) - self._flagged_posts - self._false_positives
        missed = max(0, missed)
        # misinformation spread = proportion of fake posts that slipped through
        fake_count = sum(
            1 for p, _ in self._episode[:self._step_idx + 1]
            if p["truth_score"] < self._task_cfg.fake_threshold
        )
        self._spread_rate = (
            self._false_negatives / max(fake_count, 1)
            if fake_count > 0 else 0.0
        )
        self._spread_rate = min(1.0, self._spread_rate)

        self._step_idx += 1

        # Check done
        if self._step_idx >= len(self._episode):
            self._done = True

        obs = self._make_observation()

        info = {
            "episode_id": self._episode_id,
            "step": self._step_idx,
            "result": result_str,
            "false_positive": is_fp,
            "false_negative": is_fn,
            "cumulative_reward": round(self._total_reward, 4),
            "spread_rate": round(self._spread_rate, 4),
        }

        if self._done:
            max_reward = self._max_possible_reward()
            final_score = grade_episode(
                task=self._task_cfg,
                total_reward=self._total_reward,
                max_possible_reward=max_reward,
                false_positives=self._false_positives,
                false_negatives=self._false_negatives,
                unnecessary_bans=self._unnecessary_bans,
                spread_rate=self._spread_rate,
                steps=self._step_idx,
            )
            info["final_score"] = final_score
            info["correct_moderations"] = self._correct
            info["false_positives"] = self._false_positives
            info["false_negatives"] = self._false_negatives
            info["unnecessary_bans"] = self._unnecessary_bans

        return ModerationStepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    @property
    def state(self) -> ModerationState:
        """Return current episode metadata."""
        if self._task_cfg is None:
            raise RuntimeError("Call reset() first.")
        return ModerationState(
            episode_id=self._episode_id,
            step_count=self._step_idx,
            max_steps=self._task_cfg.max_steps,
            task=self._task_cfg.name,
            total_reward=round(self._total_reward, 4),
            correct_moderations=self._correct,
            false_positives=self._false_positives,
            false_negatives=self._false_negatives,
            unnecessary_bans=self._unnecessary_bans,
        )

    # ─────────────────── Helpers ────────────────────────

    def _make_observation(self) -> ModerationObservation:
        if self._done or self._step_idx >= len(self._episode):
            # Return last post info with done=True
            post_data, user_data = self._episode[-1]
        else:
            post_data, user_data = self._episode[self._step_idx]

        step_norm = self._step_idx / max(self._task_cfg.max_steps, 1)

        post = PostInfo(
            post_id=f"post_{self._episode_id[:8]}_{self._step_idx}",
            content=post_data["content"],
            truth_score=post_data["truth_score"],
            virality_score=post_data["virality_score"],
            category=post_data["category"],
            timestamp=step_norm,
        )

        user = UserInfo(
            user_id=f"user_{self._episode_id[:8]}_{self._step_idx}",
            credibility_score=user_data["credibility_score"],
            post_history_length=user_data["post_history_length"],
            prior_violations=user_data["prior_violations"],
            account_age_days=user_data["account_age_days"],
        )

        platform = PlatformMetrics(
            total_posts=self._step_idx + 1,
            flagged_posts=self._flagged_posts,
            misinformation_spread_rate=round(self._spread_rate, 4),
            false_positive_rate=round(
                self._false_positives / max(self._step_idx + 1, 1), 4
            ),
            active_users=max(100, 1000 - self._step_idx * 10),
            step_number=self._step_idx,
        )

        return ModerationObservation(
            current_post=post,
            current_user=user,
            platform_metrics=platform,
            last_action_result=self._last_result,
            last_action_reward=self._last_reward,
            episode_done=self._done,
            task_name=self._task_cfg.name.value,
            post_queue_size=max(0, len(self._episode) - self._step_idx),
        )

    def _max_possible_reward(self) -> float:
        """Upper bound: if agent played optimally every step."""
        w = self._task_cfg.reward_weights
        best_per_step = max(
            w.get("correct_flag", 0),
            w.get("correct_fact_check", 0),
            w.get("correct_ban", 0),
        )
        # Normalise same as compute_step_reward
        raw_clamped = min(1.5, best_per_step * 1.3 + 0.1)
        normalised = (raw_clamped + 1.0) / 2.5
        return normalised * self._task_cfg.max_steps
