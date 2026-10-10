# Implementation Plan: TDD + Sprints (10 days)

## TDD rules
1. **Red** → write the test, watch it fail. 2. **Green** → simplest code to pass.
3. **Refactor** with tests green.
- Unit tests never touch the network; inject fakes from `tests/fakes.py`.
- Redis in unit tests: `fakeredis`. Postgres: Docker (`docker compose up -d`) for
  integration tests, marked `@pytest.mark.integration`.
- Real Jev/model responses saved once as fixtures; live tests marked `@pytest.mark.live`.
- Coverage ≥ 80% on `app/core`.

Claude Code prompt pattern: *"Read docs/. Do Sprint N task N.x with TDD: write the listed tests, run them and show they fail, then implement."* Read every test before accepting it; the tests are your spec.

---

## Sprint 0 (Days 1–2): Setup, skeleton deploy, spikes
| # | Task |
|---|---|
| 0.1 | Repo (personal account), `pyproject.toml`, `requirements.txt`, ruff, pytest, layout |
| 0.2 | `docker-compose.yml` (Postgres 16, Redis 7); `.env.example`; `.gitignore` |
| 0.3 | FastAPI `/api/health`; Vite React app showing health status |
| 0.4 | `vercel.json`; deploy skeleton to Vercel; confirm `/api/health` and the SPA both work |
| 0.5 | Add Neon + Upstash via Vercel Marketplace; health reports db + redis ok |
| 0.6 | Alembic init; empty first migration runs locally and on Neon |
| 0.7 | GitHub Actions: ruff, pytest (with Postgres/Redis services), `npm test`, `npm run build` |
| 0.8 | **Jev spike:** decide access path; confirm `typesafe-sdk` method/field names; save 3 fixtures |
| 0.9 | **Models spike:** confirm free cheap + strong models; fill `models.yaml` |
| 0.10 | Install TypeSafe skill in Claude Code; write `data/LABELING.md` |

Tests: `test_health_ok`, `test_health_reports_db_and_redis` (integration),
`App.test.tsx: renders health status`.
**Done:** a live Vercel URL shows "API ok · DB ok · Redis ok"; CI green; fixtures saved.

---

## Sprint 1 (Days 3–4): Routing core (FR-1–FR-6, FR-9)
**1.1 Config:** `test_loads_models_yaml`, `test_unknown_model_in_route_raises`,
`test_config_version_changes_with_threshold`

**1.2 Deadline:** `test_remaining_decreases_with_fake_clock`, `test_expired_after_budget`

**1.3 Costs:** `test_cost_per_million_prices`, `test_baseline_uses_baseline_model`,
`test_savings_not_negative`

**1.4 Jev adapter + classifier:**
`test_sends_three_questions_in_one_call`, `test_parses_fixture_to_jev_decision`,
`test_expected_difficulty_from_distribution`, `test_confidence_is_min`,
`test_sdk_timeout_raises_jev_error`, `test_malformed_result_raises_jev_error`

**1.5 Policy (pure; test thoroughly):**
`test_easy_code_routes_cheap_code_chain`, `test_high_difficulty_routes_strong`,
`test_high_reasoning_prob_routes_strong`, `test_low_confidence_routes_strong_general`,
`test_jev_error_routes_strong_general`, `test_threshold_equal_counts_as_confident`,
`test_budget_exhausted_downgrades_with_reason`

**1.6 LLM client:** `test_maps_429_to_rate_limited`, `test_maps_5xx_to_server_error`,
`test_returns_token_usage`, `test_passes_timeout`

**1.7 Executor:**
`test_first_success_one_attempt`, `test_retries_rate_limit_then_succeeds`,
`test_falls_back_after_retries`, `test_client_error_skips_retry`,
`test_all_fail_raises_with_attempts`, `test_no_retry_when_deadline_too_close`,
`test_call_timeout_capped_by_remaining_budget`

**1.8 Router v1 (no storage):** `test_route_returns_answer_and_decision`,
`test_dry_run_skips_model`

**Done:** `python -m app.core.router "prompt"` works live from the laptop.

---

## Sprint 2 (Day 5): Postgres, Redis, API (FR-7, FR-8, FR-10, FR-14, FR-15, FR-18)
**2.1 Models + migration** for all tables.
**2.2 Repository (integration):** `test_saves_request_decision_calls`,
`test_failed_request_persisted`, `test_metrics_tier_split_and_savings`,
`test_metrics_p95_latency`, `test_fallback_rate`
**2.3 Redis (fakeredis):**
`test_cache_key_normalizes_whitespace_case`, `test_cache_key_includes_config_version`,
`test_cache_roundtrip_with_ttl`, `test_rate_limit_allows_then_blocks`,
`test_rate_limit_resets_next_minute`, `test_budget_counter_daily_key`
**2.4 Router v2:** `test_cache_hit_skips_jev_and_model`, `test_budget_used_for_strong_only`
**2.5 API (TestClient + dependency overrides):**
`test_route_200_schema`, `test_route_422_empty`, `test_route_429_with_retry_after`,
`test_route_502_with_attempts`, `test_requests_list_and_detail`, `test_metrics`,
`test_evals_list`

**Done:** all endpoints tested; `/api/docs` works locally and on a preview deploy.

---

## Sprint 3 (Day 6): Evaluation (FR-13, FR-17)
3.1 Write 100 labeled prompts.
3.2 Metrics (pure): `test_accuracy`, `test_confusion_matrix`,
`test_threshold_sweep_rows`, `test_dataset_rejects_bad_labels`
3.3 Runner: `test_run_persists_run_and_items`, `test_report_has_results_table`
3.4 Real dry-run eval → tune threshold and difficulty cut-off → write down *why*.
3.5 `--with-answers --limit 20` → rate quality.
**Done:** report in `reports/`; tuned values in config; results in Neon.

---

## Sprint 4 (Days 7–8): React frontend (FR-11, FR-12, FR-16)
4.1 Generate `types.ts`; `api/client.ts` with TanStack Query hooks.
4.2 Layout + routing: Chat, Dashboard, Evals; Tailwind styling.
4.3 Chat page: message list, input, loading state, error states (429, 502), decision panel.
4.4 Dashboard: metric cards + charts (tier split, reason split, cost vs baseline, latency).
4.5 Evals page: latest run summary, accuracy, confusion matrix table, sweep chart.

Tests (Vitest + RTL, API module mocked):
`DecisionPanel shows tier, reason, confidence`, `DecisionPanel shows fallback attempts`,
`Chat sends prompt and renders answer`, `Chat shows rate-limit message on 429`,
`Chat shows attempts on 502`, `Dashboard renders metric cards from data`,
`Evals renders accuracy and confusion matrix`

**Done:** full flow works locally against the local API.

---

## Sprint 5 (Day 9): Production hardening
| # | Task |
|---|---|
| 5.1 | Deploy; set all env vars in Vercel; run migrations on Neon |
| 5.2 | Test a slow/failing model path in production; confirm 502 before 60 s, never a Vercel timeout |
| 5.3 | Confirm rate limit and strong cap work in production |
| 5.4 | Check Vercel function logs show structured logs with `request_id` |
| 5.5 | Live smoke test script: `scripts/smoke.py <url>` hits health, route (dry run), metrics |

**Done:** public URL stable; smoke test passes.

---

## Sprint 6 (Day 10): Ship
README (problem, architecture diagram, routing explanation, results table, quality check,
limitations, how to run, live link, screenshots/GIF) · resume bullet with real numbers ·
pin repo · 5 interview Q&As: why Jev vs an LLM classifier, how you chose the threshold,
Jev failure handling, how savings were estimated, how you stayed under the 60 s limit.

---

## If you must finish in 7 days, cut these
Daily budget cap (FR-18; keep the rate limit), threshold sweep (FR-17), Evals page (FR-16;
show results in README only), dashboard reduced to 3 charts, merge Sprint 5 into Sprint 4.

## Backlog (v2)
Streaming; thumbs up/down feedback to tune routing; LLM-as-judge quality scoring;
auth + per-user quotas; OpenTelemetry tracing.
