"""
MisinformationModerationEnv – Task Definitions & Graders
=========================================================
Three tasks with deterministic, reproducible graders (scores 0.0 – 1.0).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models import ActionType, PostInfo, TaskDifficulty, UserInfo


# ─────────────────────────── Post Templates ─────────────────────────

FAKE_POSTS: List[Dict] = [
    {
        "content": "SHOCKING: Scientists prove that 5G towers cause COVID-19! Share before they delete this!",
        "truth_score": 0.02,
        "virality_score": 0.95,
        "category": "health",
    },
    {
        "content": "Government confirmed: drinking bleach cures all viruses. Mainstream media won't tell you!",
        "truth_score": 0.01,
        "virality_score": 0.88,
        "category": "health",
    },
    {
        "content": "EXPOSED: The moon landing was filmed in Hollywood. NASA admits it was fake!",
        "truth_score": 0.03,
        "virality_score": 0.72,
        "category": "science",
    },
    {
        "content": "Breaking: Vaccines contain microchips to track your location! Bill Gates admits plan!",
        "truth_score": 0.01,
        "virality_score": 0.91,
        "category": "health",
    },
    {
        "content": "Election stolen! Machines changed 15 million votes. Here is the proof they are hiding!",
        "truth_score": 0.04,
        "virality_score": 0.85,
        "category": "politics",
    },
    {
        "content": "Eating apple seeds daily cures cancer, doctors don't want you to know this secret!",
        "truth_score": 0.05,
        "virality_score": 0.78,
        "category": "health",
    },
]

CONTEXTUAL_POSTS: List[Dict] = [
    {
        "content": "New study suggests coffee may reduce risk of Alzheimer's — early research phase.",
        "truth_score": 0.62,
        "virality_score": 0.45,
        "category": "health",
    },
    {
        "content": "Unemployment dropped 0.3% last month. Opposition claims figures are manipulated.",
        "truth_score": 0.70,
        "virality_score": 0.55,
        "category": "politics",
    },
    {
        "content": "Ivermectin shows promising results in small in-vitro study — not yet peer-reviewed.",
        "truth_score": 0.45,
        "virality_score": 0.68,
        "category": "health",
    },
    {
        "content": "Celebrity claims new diet cured her autoimmune disease — no clinical evidence yet.",
        "truth_score": 0.35,
        "virality_score": 0.60,
        "category": "celebrity",
    },
    {
        "content": "Bitcoin will hit $1M by year end, anonymous crypto analyst predicts based on charts.",
        "truth_score": 0.20,
        "virality_score": 0.72,
        "category": "finance",
    },
    {
        "content": "Natural immunity from COVID-19 may last longer than vaccine immunity, some studies suggest.",
        "truth_score": 0.55,
        "virality_score": 0.65,
        "category": "health",
    },
]

TRUE_POSTS: List[Dict] = [
    {
        "content": "WHO confirms global polio eradication efforts have reduced cases by 99% since 1988.",
        "truth_score": 0.97,
        "virality_score": 0.30,
        "category": "health",
    },
    {
        "content": "NASA successfully lands Perseverance rover on Mars — live streams confirm touchdown.",
        "truth_score": 0.99,
        "virality_score": 0.80,
        "category": "science",
    },
    {
        "content": "US Federal Reserve raises interest rates by 0.25% in latest policy meeting.",
        "truth_score": 0.98,
        "virality_score": 0.42,
        "category": "finance",
    },
    {
        "content": "Meta reports Q4 earnings beat analyst expectations with $14B net income.",
        "truth_score": 0.95,
        "virality_score": 0.35,
        "category": "finance",
    },
    {
        "content": "Scientists sequence 98.7% of the human genome for the first time in history.",
        "truth_score": 0.96,
        "virality_score": 0.50,
        "category": "science",
    },
]

LOW_CREDIBILITY_USERS: List[Dict] = [
    {"credibility_score": 0.10, "post_history_length": 230, "prior_violations": 8, "account_age_days": 45},
    {"credibility_score": 0.15, "post_history_length": 180, "prior_violations": 6, "account_age_days": 30},
    {"credibility_score": 0.08, "post_history_length": 400, "prior_violations": 12, "account_age_days": 20},
]

HIGH_CREDIBILITY_USERS: List[Dict] = [
    {"credibility_score": 0.92, "post_history_length": 1500, "prior_violations": 0, "account_age_days": 1200},
    {"credibility_score": 0.88, "post_history_length": 800, "prior_violations": 1, "account_age_days": 900},
    {"credibility_score": 0.95, "post_history_length": 2000, "prior_violations": 0, "account_age_days": 1800},
]

MID_CREDIBILITY_USERS: List[Dict] = [
    {"credibility_score": 0.55, "post_history_length": 350, "prior_violations": 2, "account_age_days": 400},
    {"credibility_score": 0.48, "post_history_length": 200, "prior_violations": 3, "account_age_days": 250},
    {"credibility_score": 0.60, "post_history_length": 500, "prior_violations": 1, "account_age_days": 600},
]


# ─────────────────────────── Task Config ────────────────────────────

@dataclass
class TaskConfig:
    name: TaskDifficulty
    description: str
    max_steps: int
    post_pool: List[Dict]
    user_pool: List[Dict]
    # thresholds
    fake_threshold: float          # truth_score below this → misinformation
    credibility_threshold: float   # credibility below this → suspicious user
    ban_threshold: int             # prior_violations >= this → ban is warranted
    reward_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.reward_weights:
            self.reward_weights = {
                "correct_flag": 1.0,
                "correct_fact_check": 0.8,
                "correct_reduce": 0.6,
                "correct_ban": 1.0,
                "correct_ignore": 0.5,
                "false_positive_flag": -0.6,
                "false_positive_ban": -0.9,
                "missed_fake": -0.8,
                "unnecessary_reduce": -0.3,
            }


TASK_REGISTRY: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name=TaskDifficulty.EASY,
        description=(
            "Detect obvious fake news. Posts are clearly misinformation with very low truth scores "
            "(<0.10) and authors have poor credibility. Agent should flag or fact-check fake posts "
            "and ignore genuine ones."
        ),
        max_steps=10,
        post_pool=FAKE_POSTS + TRUE_POSTS,
        user_pool=LOW_CREDIBILITY_USERS + HIGH_CREDIBILITY_USERS,
        fake_threshold=0.20,
        credibility_threshold=0.30,
        ban_threshold=10,
    ),
    "medium": TaskConfig(
        name=TaskDifficulty.MEDIUM,
        description=(
            "Handle contextual misinformation. Posts have ambiguous truth scores (0.3-0.7) and users "
            "have mixed credibility. Agent must weigh multiple signals and choose the most appropriate "
            "moderation action without over-moderating."
        ),
        max_steps=15,
        post_pool=CONTEXTUAL_POSTS + TRUE_POSTS + FAKE_POSTS[:2],
        user_pool=MID_CREDIBILITY_USERS + HIGH_CREDIBILITY_USERS + LOW_CREDIBILITY_USERS[:1],
        fake_threshold=0.40,
        credibility_threshold=0.40,
        ban_threshold=8,
    ),
    "hard": TaskConfig(
        name=TaskDifficulty.HARD,
        description=(
            "Minimise misinformation spread over time while avoiding false positives. All post types "
            "are mixed in. Agent must maintain low false-positive rates AND keep misinformation spread "
            "below 20% over a 20-step episode. Both metrics contribute to the final score."
        ),
        max_steps=20,
        post_pool=FAKE_POSTS + CONTEXTUAL_POSTS + TRUE_POSTS,
        user_pool=LOW_CREDIBILITY_USERS + MID_CREDIBILITY_USERS + HIGH_CREDIBILITY_USERS,
        fake_threshold=0.35,
        credibility_threshold=0.35,
        ban_threshold=6,
        reward_weights={
            "correct_flag": 0.9,
            "correct_fact_check": 1.0,   # prefer fact-checking in hard mode
            "correct_reduce": 0.7,
            "correct_ban": 0.8,
            "correct_ignore": 0.6,
            "false_positive_flag": -0.7,
            "false_positive_ban": -1.0,
            "missed_fake": -0.9,
            "unnecessary_reduce": -0.4,
        },
    ),
}


# ─────────────────────────── Episode Generator ──────────────────────

def generate_episode(task: TaskConfig, rng: random.Random) -> List[Tuple[Dict, Dict]]:
    """Return a shuffled list of (post, user) dicts for an episode."""
    posts = rng.choices(task.post_pool, k=task.max_steps)
    users = rng.choices(task.user_pool, k=task.max_steps)
    return list(zip(posts, users))


# ─────────────────────────── Reward Calculator ──────────────────────

def compute_step_reward(
    action: ActionType,
    post: Dict,
    user: Dict,
    task: TaskConfig,
    spread_rate: float,
) -> Tuple[float, str, bool, bool]:
    """
    Compute dense, normalised step reward ∈ [0, 1].

    Returns (raw_reward, result_str, is_false_positive, is_false_negative).
    """
    w = task.reward_weights
    truth = post["truth_score"]
    cred = user["credibility_score"]
    violations = user["prior_violations"]
    virality = post["virality_score"]

    is_fake = truth < task.fake_threshold
    suspicious_user = cred < task.credibility_threshold
    ban_warranted = violations >= task.ban_threshold

    is_fp = False
    is_fn = False
    raw = 0.0
    result = ""

    if action == ActionType.FLAG_CONTENT:
        if is_fake:
            # Bonus for catching high-virality fakes
            raw = w["correct_flag"] * (0.7 + 0.3 * virality)
            result = "Correct: flagged misinformation"
        else:
            raw = w["false_positive_flag"]
            is_fp = True
            result = "False positive: flagged legitimate content"

    elif action == ActionType.FACT_CHECK:
        if is_fake:
            # Fact-checking is slightly weaker than flagging but more nuanced
            raw = w["correct_fact_check"] * (0.6 + 0.4 * virality)
            result = "Correct: fact-checked misinformation"
        elif 0.35 <= truth <= 0.65:
            # Contextual content — fact-checking is appropriate
            raw = w["correct_fact_check"] * 0.5
            result = "Appropriate: fact-checked ambiguous content"
        else:
            raw = w["false_positive_flag"] * 0.5
            is_fp = True
            result = "Unnecessary: fact-checked clearly true content"

    elif action == ActionType.REDUCE_VISIBILITY:
        if is_fake and virality > 0.6:
            raw = w["correct_reduce"] * virality
            result = "Correct: reduced visibility of viral misinformation"
        elif is_fake:
            raw = w["correct_reduce"] * 0.4
            result = "Partially correct: reduced visibility (low virality)"
        else:
            raw = w["unnecessary_reduce"]
            is_fp = True
            result = "Unnecessary: reduced visibility of legitimate content"

    elif action == ActionType.BAN_USER:
        if ban_warranted and suspicious_user:
            raw = w["correct_ban"]
            result = "Correct: banned repeat offender"
        elif ban_warranted:
            raw = w["correct_ban"] * 0.5
            result = "Partial: ban justified by violations but user credibility mixed"
        else:
            raw = w["false_positive_ban"]
            is_fp = True
            result = "False positive: unjustly banned user"

    elif action == ActionType.IGNORE:
        if not is_fake:
            raw = w["correct_ignore"]
            result = "Correct: ignored legitimate content"
        else:
            raw = w["missed_fake"]
            is_fn = True
            # Penalise more for missing high-virality fakes
            raw = w["missed_fake"] * (0.5 + 0.5 * virality)
            result = f"Missed: ignored misinformation (virality={virality:.2f})"

    # Spread penalty bonus: reward for keeping spread low
    if spread_rate < 0.15:
        raw += 0.1
    elif spread_rate > 0.40:
        raw -= 0.1

    # Normalise to [0, 1]
    raw_clamped = max(-1.0, min(1.5, raw))
    normalised = (raw_clamped + 1.0) / 2.5   # maps [-1,1.5] → [0,1]
    return round(normalised, 4), result, is_fp, is_fn


# ─────────────────────────── Episode Grader ─────────────────────────

def grade_episode(
    task: TaskConfig,
    total_reward: float,
    max_possible_reward: float,
    false_positives: int,
    false_negatives: int,
    unnecessary_bans: int,
    spread_rate: float,
    steps: int,
) -> float:
    """
    Deterministic episode grader → score ∈ [0.0, 1.0].

    Combines reward efficiency with quality metrics.
    """
    if steps == 0:
        return 0.0

    # Reward efficiency
    reward_score = min(1.0, total_reward / max(max_possible_reward, 1e-6))
    reward_score = max(0.0, reward_score)

    # Quality penalties
    fp_penalty = min(1.0, (false_positives + unnecessary_bans) / max(steps, 1))
    fn_penalty = min(1.0, false_negatives / max(steps, 1))

    if task.name == TaskDifficulty.EASY:
        score = 0.7 * reward_score - 0.15 * fp_penalty - 0.15 * fn_penalty

    elif task.name == TaskDifficulty.MEDIUM:
        score = 0.6 * reward_score - 0.20 * fp_penalty - 0.20 * fn_penalty

    else:  # HARD — also penalise spread
        spread_penalty = max(0.0, spread_rate - 0.20)   # penalise if > 20%
        score = (
            0.5 * reward_score
            - 0.20 * fp_penalty
            - 0.15 * fn_penalty
            - 0.15 * spread_penalty
        )

    return round(max(0.0, min(1.0, score)), 4)
