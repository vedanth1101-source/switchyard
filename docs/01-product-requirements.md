# Product Requirements Document: Switchyard

**Owner:** Vedanth M.S. · **Status:** Draft v2 · **Timeline:** 10 days · **Budget:** $0

## 1. Overview
Switchyard is an LLM request router. For each prompt, it asks TypeSafe's Jev decision
model to classify the **task type** (code / math / writing / general) and **difficulty**,
then sends the prompt to the cheapest model likely to answer it well. Easy prompts go to
a cheap model; hard or uncertain ones go to a strong model. A React web app shows every
routing decision and a dashboard of savings, latency, and reliability.

## 2. Problem
Apps often send every prompt to one expensive model, even trivial ones. Using an LLM to
decide routing is itself slow and costly. Jev is built for fast, cheap, structured
decisions like this.

## 3. Goals
- G1: Route prompts automatically by task type and difficulty.
- G2: Show measured cost savings vs. always using the strong model, with a quality check.
- G3: Stay reliable through low-confidence decisions, Jev failures, and model errors.
- G4: Explain every decision in the UI.
- G5: Run entirely on free tiers, including when the demo is public.

## 4. Non-goals
User accounts and login; streaming responses; multi-tenant support; production SLAs;
training or fine-tuning models.

## 5. Users
- **Demo user / interviewer:** chats with the router and explores the dashboard.
- **Developer:** calls `POST /api/route` and gets the answer plus metadata.
- **Builder (me):** runs evaluations and tunes thresholds.

## 6. User stories
- US-1: I type a prompt and get an answer from an automatically chosen model.
- US-2: I see why that model was chosen: task type, difficulty, confidence, tier, reason.
- US-3: As a developer, I call `POST /api/route` and get the answer plus metadata.
- US-4: I see a dashboard of routing split, estimated cost saved, latency, fallback and
  cache rates.
- US-5: As the builder, I run one command to evaluate routing accuracy and get a report;
  results also appear on an Evals page.
- US-6: My request still succeeds if one model is down or rate-limited.

## 7. Functional requirements
| ID | Requirement | Priority |
|---|---|---|
| FR-1 | One Jev call per prompt: task type (Choice), difficulty (Score, 5 levels), needs multi-step reasoning (Noul) | Must |
| FR-2 | Map decision to tier (cheap/strong) and model chain from `models.yaml` | Must |
| FR-3 | If Jev confidence < threshold → strong tier (low-confidence fallback) | Must |
| FR-4 | If Jev fails or times out → safe default (strong general) | Must |
| FR-5 | Call the selected model and return its answer | Must |
| FR-6 | Retry rate-limit/server errors, then fall back to next model, all within a 50 s request budget | Must |
| FR-7 | Cache decision + answer in Redis for identical prompts (TTL 24 h) | Must |
| FR-8 | Persist requests, decisions, model calls, eval runs in Postgres | Must |
| FR-9 | Estimate each request's cost and the baseline cost (always-strong) | Must |
| FR-10 | REST API under `/api` (FastAPI) | Must |
| FR-11 | React chat page with a decision panel per answer | Must |
| FR-12 | React dashboard: tier/task/reason split, est. cost vs baseline, latency p50/p95, fallback rate, cache hit rate | Must |
| FR-13 | Evaluation CLI over a labeled dataset; writes report files and stores results in Postgres | Must |
| FR-14 | Per-IP rate limit (protects free credits on the public demo) | Must |
| FR-15 | Dry-run mode: decision only, no model call | Should |
| FR-16 | React Evals page showing latest eval run | Should |
| FR-17 | Confidence-threshold sweep in the eval report | Should |
| FR-18 | Daily strong-model budget cap; degrade to cheap with reason `BUDGET` | Should |

## 8. Success metrics (targets; report real results honestly)
| Metric | Target |
|---|---|
| Task-type accuracy (eval set) | ≥ 85% |
| Tier accuracy (eval set) | ≥ 80% |
| Prompts routed to cheap tier | ≥ 50% |
| Estimated cost reduction vs always-strong | ≥ 40% |
| Jev decision latency p95 | < 800 ms |
| Requests succeeding despite a model failure | ≥ 95% |
| Requests finishing within Vercel's 60 s limit | 100% (return error, never time out) |
| Money spent | $0 |

Quality check: manually rate 20 cheap-routed answers (good / acceptable / bad).

## 9. Risks
| Risk | Mitigation |
|---|---|
| No free Jev access path | Day 1 spike; fallback is a few cents of TypeSafe balance |
| Vercel Hobby 60 s function limit | Request time budget, per-call timeouts, eval runs locally |
| Public demo drains free credits | Per-IP rate limit, daily strong-model cap, cache |
| Free model rate limits | Retries, fallback chains, cache, dry-run eval |
| Savings are hypothetical (free models) | Estimate with list prices; state clearly in README |
| Eval labels are my own judgment | Write `LABELING.md` rules before labeling |
| New stack slows me down | 10-day plan; 7-day cut list in implementation plan |

## 10. Release plan
Days 1–2 setup + deploy skeleton + Jev spike → Days 3–4 routing core → Day 5 storage,
cache, API → Day 6 evaluation → Days 7–8 React frontend → Day 9 production hardening →
Day 10 README, resume, buffer.
