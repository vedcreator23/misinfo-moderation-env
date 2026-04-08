# MisinformationModerationEnv

An **OpenEnv-compliant reinforcement learning environment** that simulates a **real-world social media content moderation system**. An AI agent reviews posts, analyses truth scores, virality, and user credibility, then chooses the most appropriate moderation action.

---

## 🔍 Environment Description

Social media platforms face a firehose of content — from verifiable news to blatant conspiracy theories. This environment models the moderation pipeline where an agent must:

- Evaluate individual posts for misinformation
- Consider user credibility and violation history
- Monitor platform-wide spread metrics
- Balance aggressive moderation against false positives
- Optimise across a full episode trajectory

---

## 📐 Observation Space

Each observation (`ModerationObservation`) includes:

| Field | Type | Description |
|---|---|---|
| `current_post.content` | str | The post text to review |
| `current_post.truth_score` | float [0,1] | Ground-truth accuracy (0=fake, 1=true) |
| `current_post.virality_score` | float [0,1] | Normalised spread speed |
| `current_post.category` | str | Topic: health/politics/science/finance/celebrity |
| `current_user.credibility_score` | float [0,1] | Author historical accuracy |
| `current_user.prior_violations` | int | Past moderation actions |
| `current_user.account_age_days` | int | Platform tenure |
| `platform_metrics.misinformation_spread_rate` | float [0,1] | Proportion of fakes that slipped through |
| `platform_metrics.false_positive_rate` | float [0,1] | Proportion of legitimate content flagged |
| `post_queue_size` | int | Remaining posts in episode |

---

## 🎮 Action Space

Each step accepts a `ModerationAction` with one of five action types:

| Action | When to use |
|---|---|
| `flag_content` | Clear-cut misinformation — immediate removal |
| `fact_check` | Ambiguous or contextual claims — send for review |
| `reduce_visibility` | Viral but uncertain content — throttle reach |
| `ban_user` | Repeat offenders with low credibility (use sparingly) |
| `ignore` | Clearly legitimate content — no action needed |

---

## 📋 Tasks

### Easy — *Detect Obvious Fake News*
Posts have very low truth scores (< 0.10) and authors have poor credibility. The signal is clear. Agent should flag/fact-check fakes and ignore genuine content.
- **Max steps:** 10

### Medium — *Handle Contextual Misinformation*
Posts have ambiguous truth scores (0.3–0.7) and mixed-credibility authors. Agent must weigh all signals without over-moderating.
- **Max steps:** 15

### Hard — *Minimise Spread, Avoid False Positives*
All post types are mixed. Agent must keep misinformation spread below 20% while maintaining a low false-positive rate across a full 20-step episode.
- **Max steps:** 20

---

## 🏆 Reward Function

Dense reward per step, normalised to **[0.0, 1.0]**:

| Situation | Signal |
|---|---|
| Correctly flagged misinformation | +1.0 (scaled by virality) |
| Correct fact-check | +0.8 |
| Correctly reduced visibility | +0.6 |
| Correctly banned repeat offender | +1.0 |
| Correctly ignored legitimate content | +0.5 |
| False positive (flagged real content) | −0.6 |
| Unjust ban | −0.9 |
| Missed misinformation | −0.8 (scaled by virality) |
| Low spread bonus | +0.1 |
| High spread penalty | −0.1 |

Episode score (returned in `info["final_score"]`) is a weighted combination of reward efficiency and quality metrics (FP/FN rates, spread rate).

---

## ⚙️ Setup & Usage

### Prerequisites
```bash
pip install fastapi uvicorn pydantic httpx openai
```

### Run locally (no Docker)
```bash
cd /path/to/code
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t misinfo-moderation-env .
docker run -p 7860:7860 misinfo-moderation-env
```

### Use the environment programmatically
```python
from client import MisinformationModerationEnv
from models import ModerationAction, ActionType

with MisinformationModerationEnv(base_url="http://localhost:7860") as env:
    obs = env.reset(task="easy")
    print(obs.current_post.content)
    result = env.step(ModerationAction(action_type=ActionType.FLAG_CONTENT))
    print(result.reward)
```

### Run the baseline inference script
```bash
export HF_TOKEN=<your_token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export BASE_URL=http://localhost:7860

python inference.py
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks with descriptions |
| POST | `/reset` | Start new episode `{"task": "easy"}` |
| POST | `/step` | Submit action `{"action_type": "flag_content", ...}` |
| GET | `/state` | Current episode metadata |

---

## 📦 Project Structure

```
.
├── models.py                # Pydantic models (Action, Observation, State)
├── tasks.py                 # Task configs & graders
├── client.py                # HTTP client
├── __init__.py              # Package exports
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package metadata
├── Dockerfile               # Container definition
├── .dockerignore
├── README.md
└── server/
    ├── environment.py       # Core environment logic
    ├── app.py               # FastAPI server
    └── requirements.txt     # Server dependencies
```

---

## 📊 Baseline Scores

| Task | Model | Score |
|---|---|---|
| easy | Qwen2.5-72B-Instruct | ~0.72 |
| medium | Qwen2.5-72B-Instruct | ~0.55 |
| hard | Qwen2.5-72B-Instruct | ~0.38 |

---

## 🚀 Deploy to HuggingFace Spaces

```bash
pip install huggingface_hub
huggingface-cli login
openenv push --repo-id RV987654321/misinfo-moderation-env
```

Or manually: create a new Space with **Docker** SDK, push to [RV987654321/misinfo-moderation-env](https://huggingface.co/RV987654321/misinfo-moderation-env), and the `Dockerfile` handles the rest.

---

## 🔗 Links

- **GitHub Repository**: [vedcreator23/misinfo-moderation-env](https://github.com/vedcreator23)
- **Hugging Face Space**: [RV987654321/misinfo-moderation-env](https://huggingface.co/RV987654321)

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (inference) | — | HuggingFace / API key |
| `API_BASE_URL` | No | HF router | LLM endpoint |
| `MODEL_NAME` | No | Qwen2.5-72B | Model identifier |
| `BASE_URL` | No | localhost:7860 | Env server URL |

---

## License

MIT
