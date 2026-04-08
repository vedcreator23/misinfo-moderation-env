"""
MisinformationModerationEnv – Data Models
==========================================
OpenEnv-compliant Pydantic models for Actions, Observations, State and Rewards.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────── Enumerations ───────────────────────────

class ActionType(str, Enum):
    FLAG_CONTENT = "flag_content"
    FACT_CHECK = "fact_check"
    REDUCE_VISIBILITY = "reduce_visibility"
    BAN_USER = "ban_user"
    IGNORE = "ignore"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ─────────────────────────── Sub-models ─────────────────────────────

class PostInfo(BaseModel):
    post_id: str
    content: str
    truth_score: float = Field(ge=0.0, le=1.0, description="Ground truth score (0=fake, 1=true)")
    virality_score: float = Field(ge=0.0, le=1.0, description="Normalised viral spread score")
    category: str = Field(description="Topic category: health/politics/science/celebrity/finance")
    timestamp: float = Field(description="Relative time in episode (0.0 – 1.0)")


class UserInfo(BaseModel):
    user_id: str
    credibility_score: float = Field(ge=0.0, le=1.0, description="Historical post accuracy")
    post_history_length: int = Field(ge=0, description="Total posts on platform")
    prior_violations: int = Field(ge=0, description="Number of past moderation actions")
    account_age_days: int = Field(ge=0)


class PlatformMetrics(BaseModel):
    total_posts: int
    flagged_posts: int
    misinformation_spread_rate: float = Field(ge=0.0, le=1.0)
    false_positive_rate: float = Field(ge=0.0, le=1.0)
    active_users: int
    step_number: int


# ─────────────────────────── Action ─────────────────────────────────

class ModerationAction(BaseModel):
    """OpenEnv Action model – one moderation decision per step."""
    action_type: ActionType
    post_id: Optional[str] = None   # required for flag/fact-check/reduce_visibility
    user_id: Optional[str] = None   # required for ban_user
    reasoning: Optional[str] = None # optional chain-of-thought (ignored by env)


# ─────────────────────────── Observation ────────────────────────────

class ModerationObservation(BaseModel):
    """OpenEnv Observation model – returned after every step / reset."""
    # Current post under review
    current_post: PostInfo
    # Author of current post
    current_user: UserInfo
    # Platform-wide snapshot
    platform_metrics: PlatformMetrics
    # Recent action results
    last_action_result: Optional[str] = None
    last_action_reward: Optional[float] = None
    # Episode meta
    episode_done: bool = False
    task_name: str = "easy"
    # Context: last 3 posts seen this episode
    post_queue_size: int = 0


# ─────────────────────────── State ──────────────────────────────────

class ModerationState(BaseModel):
    """OpenEnv State – episode-level metadata."""
    episode_id: str
    step_count: int
    max_steps: int
    task: TaskDifficulty
    total_reward: float
    correct_moderations: int
    false_positives: int
    false_negatives: int
    unnecessary_bans: int


# ─────────────────────────── StepResult ─────────────────────────────

class ModerationStepResult(BaseModel):
    """Full result returned by step()."""
    observation: ModerationObservation
    reward: float
    done: bool
    info: Dict = Field(default_factory=dict)
