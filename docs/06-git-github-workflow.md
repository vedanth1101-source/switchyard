# Git & GitHub Workflow: Switchyard

## 1. One-time setup
```bash
# Identity (use the email on your GitHub account, or your GitHub noreply email
# from Settings → Emails if you want to keep your real email private)
git config --global user.name "Vedanth M.S."
git config --global user.email "<your-github-email>"
git config --global init.defaultBranch main

# GitHub CLI (makes issues, PRs, releases easy from the terminal)
gh auth login

# Create the repo (public, personal account)
mkdir switchyard && cd switchyard
git init
# add .gitignore, README.md, docs/ first
git add . && git commit -m "chore: initial project docs and gitignore"
gh repo create switchyard --public --source=. --push
```

`.gitignore` must include at minimum:
```
.env
.env.*
!.env.example
__pycache__/
.venv/
*.pyc
.pytest_cache/
.coverage
node_modules/
web/dist/
.vercel/
```

## 2. Repository settings (GitHub → Settings)
- **Description + topics:** `llm`, `model-routing`, `fastapi`, `react`, `postgres`,
  `redis`, `vercel`, `jev`. Add the live demo URL to the "Website" field.
- **Security → Secret scanning + push protection: ON.** Blocks pushes containing API keys.
- **Rules → Ruleset for `main`:** require a pull request, require CI status checks to
  pass, block force pushes and deletion. Don't require approvals (you can't approve
  your own PR).
- **General → Pull requests:** allow **Rebase and merge** only; enable
  "Automatically delete head branches".

Why rebase-and-merge: it keeps `main` linear and clean, *and* keeps your individual
test → implement → refactor commits, so your TDD process is visible in the history.

## 3. Planning in GitHub
- **Milestones:** one per sprint (`Sprint 0: Setup` … `Sprint 6: Ship`), with due dates.
- **Issues:** one per task in the implementation plan (e.g. "1.5 Policy: routing rules").
  Paste the task's test list into the issue as a checklist.
- **Labels:** `feat`, `test`, `bug`, `docs`, `chore`, plus areas `core`, `api`, `db`,
  `cache`, `web`, `eval`, `ci`.
- **Project board** (GitHub Projects): columns `Todo`, `In progress`, `Done`; issues
  move automatically when PRs close them.

## 4. Branches
GitHub Flow: `main` is always deployable; all work happens on short-lived branches.

| Branch | Example |
|---|---|
| `feat/<sprint>-<topic>` | `feat/s1-policy` |
| `fix/<topic>` | `fix/executor-deadline` |
| `test/<topic>` | `test/metrics-edge-cases` |
| `docs/<topic>` | `docs/readme-results` |
| `chore/<topic>` | `chore/ci-postgres-service` |

One branch per issue. Merge within a day where possible.

## 5. Commit messages (Conventional Commits)
Format: `type(scope): short summary in present tense`

| Type | Use for |
|---|---|
| `feat` | New behavior |
| `fix` | Bug fix |
| `test` | Adding/changing tests |
| `refactor` | Code change with no behavior change |
| `docs` | Documentation |
| `chore` | Tooling, deps, config |
| `ci` | GitHub Actions |
| `perf` | Performance |

TDD rhythm example for one feature:
```
test(policy): add failing tests for low-confidence fallback
feat(policy): route low-confidence decisions to strong general model (FR-3)
refactor(policy): extract threshold comparison helper
```
Rules: small commits; each commit should pass tests (except the deliberate red test
commit); never commit commented-out code or debug prints; mention FR ids where relevant.

Claude Code can write commits for you. Read each diff before it commits, and keep its
messages in this format. If it adds a co-author line crediting Claude, keep it; being
transparent about AI assistance is a positive in interviews, as long as you can explain
the code.

## 6. Pull requests
Every change reaches `main` through a PR, even working solo. Create
`.github/pull_request_template.md`:
```markdown
## What
<!-- one or two sentences -->

## Why
Closes #<issue>  ·  FR: <ids>

## How tested
- [ ] Unit tests added/updated
- [ ] `pytest` and `npm test` pass locally
- [ ] Checked Vercel preview deploy (if UI/API changed)

## Screenshots
<!-- for UI changes -->
```
Flow: push branch → `gh pr create --fill` → CI runs + Vercel posts a preview URL →
check preview → rebase-and-merge → branch auto-deletes.

## 7. Daily routine
```bash
git switch main && git pull
git switch -c feat/s1-policy
# Red: write tests
git add tests/ && git commit -m "test(policy): add failing tests for routing rules"
# Green: implement
git add app/ && git commit -m "feat(policy): implement tier and model selection (FR-2)"
# Refactor
git commit -am "refactor(policy): simplify reason selection"
git push -u origin feat/s1-policy
gh pr create --fill
# after CI is green:
gh pr merge --rebase
git switch main && git pull
```

## 8. Versions and releases
Semantic versioning: `MAJOR.MINOR.PATCH`. Before 1.0, each sprint is a minor version.

| Version | After | Highlights |
|---|---|---|
| v0.1.0 | Sprint 0 | Skeleton deployed on Vercel, DB + Redis connected |
| v0.2.0 | Sprint 1 | Routing core with Jev, policy, retries, fallback |
| v0.3.0 | Sprint 2 | Postgres, Redis cache, rate limit, REST API |
| v0.4.0 | Sprint 3 | Evaluation harness + first results |
| v0.5.0 | Sprint 4 | React chat, dashboard, evals pages |
| v0.6.0 | Sprint 5 | Production hardening |
| v1.0.0 | Sprint 6 | Public launch with README results |

Bug fixes between sprints bump PATCH (e.g. v0.3.1).

Release at the end of each sprint:
```bash
git switch main && git pull
# update CHANGELOG.md, commit: "docs: changelog for v0.2.0"
git tag -a v0.2.0 -m "v0.2.0: routing core"
git push origin v0.2.0
gh release create v0.2.0 --title "v0.2.0: Routing core" --generate-notes
```
Close the sprint's milestone after the release.

`CHANGELOG.md` format:
```markdown
## [0.2.0] - 2026-10-14
### Added
- Jev classifier with task type, difficulty, and reasoning questions (FR-1)
- Routing policy with low-confidence fallback (FR-2, FR-3)
### Fixed
- …
```

## 9. Secrets safety
- Never commit `.env`. Commit `.env.example` with empty values.
- Keys live in local `.env`, Vercel environment variables, and GitHub Actions secrets.
- If a key is ever pushed: **revoke and regenerate it immediately**. Deleting the commit
  is not enough; the key must be treated as leaked.

## 10. Making it count on your profile
- Pin `switchyard` on your profile.
- README top: one-line pitch, live demo link, CI badge, screenshot/GIF, results table.
- Optional: a profile README (`<username>/<username>` repo) linking to the project.
- Commit when you've done real work. A steady history of meaningful commits, PRs, and
  releases reads far better than a padded graph.
