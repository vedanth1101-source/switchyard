# Switchyard — Project Documentation

**Owner:** Vedanth M.S.  
**Status:** Draft v2  
**Timeline:** 10 days  
**Budget:** $0

Switchyard is an LLM request router. It uses TypeSafe Jev to classify prompts by task type and difficulty, then routes each prompt to the cheapest model likely to answer it well.

## Documents

1. [`01-product-requirements.md`](docs/01-product-requirements.md) — Product Requirements Document (PRD)
2. [`02-technical-requirements.md`](docs/02-technical-requirements.md) — Technical Requirements Document (TRD)
3. [`03-project-flow.md`](docs/03-project-flow.md) — Request, policy, execution, evaluation, and frontend flows
4. [`04-data-schema-and-api.md`](docs/04-data-schema-and-api.md) — Postgres, Redis, and API schema
5. [`05-implementation-plan.md`](docs/05-implementation-plan.md) — Ten-day TDD implementation plan
6. [`06-git-github-workflow.md`](docs/06-git-github-workflow.md) — Git, GitHub, PR, CI, and release workflow
7. [`07-jev-playground-spike.md`](docs/07-jev-playground-spike.md) — Initial manual Jev Playground observations

## Important verification note

The PRD and technical documents describe the intended design. Confirm current SDK method names, response fields, confidence availability, provider access, free-tier limits, and Vercel deployment behavior during Sprint 0 before treating those assumptions as verified.
