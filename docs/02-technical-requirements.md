# Technical Requirements Document: Switchyard

## 1. Stack
| Layer | Choice | Notes |
|---|---|---|
| Backend | Python 3.11+, FastAPI, Pydantic v2 | Runs as a Vercel Python function |
| ORM / migrations | SQLModel + Alembic | |
| Database | Postgres on Neon (Vercel Marketplace, free) | Use the **pooled** connection string |
| Cache / counters | Redis on Upstash (Vercel Marketplace, free) via `redis-py` (`rediss://` URL) | Cache, rate limit, budget counter |
| Jev | `typesafe-sdk` (official) | Wrapped in `jev_client.py` |
| LLM calls | `openai` SDK with custom `base_url` | Vercel AI Gateway / GitHub Models / Azure |
| Frontend | React 18 + Vite + TypeScript, Tailwind, React Router, TanStack Query, Recharts | Static build served by Vercel |
| API types | `openapi-typescript` generates `web/src/api/types.ts` from FastAPI's OpenAPI | Keeps frontend and backend in sync |
| Backend tests | pytest, pytest-cov, fakeredis | Postgres in Docker for integration tests |
| Frontend tests | Vitest + React Testing Library | |
| Lint | ruff (Python), ESLint + Prettier (TS) | |
| CI | GitHub Actions with Postgres + Redis service containers | |
| Hosting | One Vercel project (Hobby): static frontend + FastAPI function | Personal GitHub repo, not an org repo |

## 2. Architecture
```text
Browser (React SPA) ──/api/*──> Vercel Python function (FastAPI)
                                   │
                                   ▼
                       Rate limiter (Redis) → Router (orchestrator)
     ┌──────────┬─────────────┬────────────┬────────────┬───────────┐
     ▼          ▼             ▼            ▼            ▼           ▼
  Cache      Classifier     Policy      Executor     Budget      Repository
 (Redis)    (Jev adapter)  (pure fn)   (retry +     (Redis)     (Postgres)
               │                        fallback +
               ▼                        deadline)
        Jev (typesafe-sdk)                 │
                                LLM providers (OpenAI-compatible)

Laptop: eval CLI ──> same core code ──> Postgres (local or Neon)
```

## 3. Components
| Module | Responsibility | Pure? |
|---|---|---|
| `core/settings.py` | Env settings (`pydantic-settings`) | – |
| `core/model_config.py` | Load/validate `models.yaml`; `config_version` hash | – |
| `core/deadline.py` | Request time budget: `remaining()`, `expired()` | Yes (injectable clock) |
| `core/jev_client.py` | Only file that touches `typesafe-sdk`; returns `JevDecision` | No |
| `core/classifier.py` | Question definitions; expected difficulty; overall confidence | Yes |
| `core/policy.py` | `decide_route(decision, cfg, budget_ok) -> RouteDecision` | Yes |
| `core/llm_client.py` | `complete(model, prompt, timeout) -> ModelResult`; maps errors | No |
| `core/executor.py` | Chain + retries + backoff within deadline | Logic pure, clients injected |
| `core/costs.py` | Cost, baseline, savings | Yes |
| `core/router.py` | Orchestrates rate-limit → cache → classify → policy → execute → persist | No |
| `infra/db.py`, `db/models.py`, `db/repo.py` | Engine, tables, queries | No |
| `infra/redis.py`, `infra/cache.py`, `infra/ratelimit.py`, `infra/budget.py` | Redis helpers | No |
| `routes/*.py` | FastAPI endpoints | No |
| `eval/*` | Dataset loader, runner, metrics, sweep, report | Metrics pure |
| `web/` | Chat, Dashboard, Evals pages | – |

Rule: every external service sits behind a small `Protocol`; tests inject fakes.

## 4. Jev usage (confirm exact names in Sprint 0)
Sketch based on the SDK's published examples:
```python
from typesafe_sdk import TypeSafeClient, Choice, Score, Noul

client = TypeSafeClient(model=settings.jev_model, timeout=3)  # keep SDK retries low
result = client.system_one(
    state=prompt,
    questions={
        "task_type": Choice(
            instructions="What kind of task is this prompt asking for?",
            criteria={"code": None, "math": None, "writing": None, "general": None},
        ),
        "difficulty": Score(
            instructions="How hard is it to answer this well?",
            criteria=["trivial", "easy", "moderate", "hard", "expert"],
        ),
        "needs_reasoning": Noul(
            instructions="Does a good answer require multi-step reasoning?",
        ),
    },
)
```
Map Score levels to 1–5. `difficulty` = expected value; `confidence` = min of the task-type and difficulty confidences.

## 5. Routing policy (initial values; tuned in Sprint 3)
```text
if rate limit exceeded:                 HTTP 429 (no Jev call)
if use_cache and cache hit:             return cached, cache_hit = true (original reason kept; S2.14)
if jev failed:                          STRONG general, reason = JEV_ERROR
elif confidence < threshold:            STRONG general, reason = LOW_CONFIDENCE
elif difficulty >= 3.5 or p_reasoning >= 0.6: STRONG [task], reason = HARD
else:                                   CHEAP [task],  reason = EASY
if tier == STRONG and daily cap reached: CHEAP [task], reason = BUDGET
```

## 6. Reliability and time budget (Vercel Hobby: max 60 s per function)
- `vercel.json` sets `maxDuration: 60`. App-level request budget: **50 s**.
- Jev: timeout 3 s, SDK retries ≤ 1. Failure → JEV_ERROR route.
- Model call timeout = `min(20 s, deadline.remaining() - 2 s)`.
- Retry 429/5xx/timeout up to 2× with backoff (1 s, 2 s) **only if time remains**;
  other 4xx → next model immediately.
- Budget exhausted or chain fails → HTTP 502 with attempt log; request still persisted.
- Eval runs from the laptop, never on Vercel.

## 7. Non-functional requirements
| Area | Requirement |
|---|---|
| Latency | Jev p95 < 800 ms; routing overhead (excl. model) < 1 s |
| Cost | $0; never buy Vercel AI Gateway credits (ends free tier) |
| Abuse | Per-IP limit 10 req/min on `/api/route`; daily strong cap 200 |
| Security | Secrets only in Vercel env vars / local `.env`; `.env` git-ignored |
| Privacy | Prompts stored; UI notice: don't enter personal data |
| Serverless DB | Neon pooled URL; SQLAlchemy `NullPool`; `pool_pre_ping=True` |
| Tests | ≥ 80% coverage on `app/core`; unit tests offline in < 10 s |
| Observability | `request_id` on every response; structured JSON logs (visible in Vercel logs) |

## 8. Configuration
`.env` / Vercel environment variables
```dotenv
TYPESAFE_API_KEY=            # or gateway key, per Sprint 0 decision
TYPESAFE_BASE_URL=           # only if not using the default
JEV_MODEL=jev-1.13.0         # pin a version; jev-latest moves
AI_GATEWAY_API_KEY=
GITHUB_TOKEN=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
DATABASE_URL=postgresql+psycopg://...   # Neon pooled URL in prod
REDIS_URL=rediss://...                  # Upstash in prod
CONFIDENCE_THRESHOLD=0.6
CACHE_TTL_SECONDS=86400
CACHE_NAMESPACE=                        # e.g. production; cache off unless set (S2.14)
CACHE_KEY_SECRET=                       # HMAC key for cache keys, >= 32 chars; cache off unless set
RATE_LIMIT_PER_MIN=10
STRONG_DAILY_CAP=200
REQUEST_BUDGET_SECONDS=50
```
`config/models.yaml`: providers, models (with list prices for estimates), routes per
task and tier, baseline model. Model IDs filled in during Sprint 0.

## 9. Repository layout
```text
switchyard/
  api/index.py                 # Vercel entrypoint: from app.main import app
  app/
    main.py
    routes/{route,requests,metrics,evals,health}.py
    core/{settings,model_config,deadline,jev_client,classifier,policy,
          llm_client,executor,costs,router}.py
    infra/{db,redis,cache,ratelimit,budget}.py
    db/{models,repo}.py
    eval/{dataset,run_eval,metrics,report}.py
  migrations/                  # Alembic
  config/models.yaml
  data/{eval_prompts.jsonl,LABELING.md}
  reports/
  tests/{unit,integration,fixtures}/  conftest.py  fakes.py
  web/
    src/api/{client.ts,types.ts}
    src/pages/{Chat,Dashboard,Evals}.tsx
    src/components/{DecisionPanel,AttemptList,MetricCard,charts/*}.tsx
    src/App.tsx  src/main.tsx
  docker-compose.yml           # local Postgres + Redis
  requirements.txt  pyproject.toml  vercel.json  .env.example
  .github/workflows/ci.yml
```

## 10. Deployment
`vercel.json` (sketch; confirm against Vercel's current Python/FastAPI docs in Sprint 0)
```json
{
  "buildCommand": "cd web && npm ci && npm run build",
  "outputDirectory": "web/dist",
  "functions": { "api/index.py": { "maxDuration": 60 } },
  "rewrites": [
    { "source": "/api/(.*)", "destination": "/api/index" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```
- Add Neon and Upstash from the Vercel Marketplace; they inject env vars.
- Run `alembic upgrade head` against Neon from the laptop (not during build).
- Final eval run points `DATABASE_URL` at Neon so the deployed Evals page shows it;
  report files are also committed to `reports/`.
