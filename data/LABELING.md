# Labeling Rules — `data/eval_prompts.jsonl`

Write and follow these rules **before** labeling any prompt (PRD §9: eval labels are the
builder's own judgment, so the rules must be fixed first). Label what an ideal router
should do, never what Jev happened to answer.

## 1. Record format

One JSON object per line (docs/04-data-schema-and-api.md, Part A):

```json
{"id": "c-001", "prompt": "Reverse a string in Python", "task_type": "code", "tier": "cheap"}
```

| Field | Values |
|---|---|
| `id` | `<prefix>-<3-digit number>`; prefix by `task_type`: `c` code, `m` math, `w` writing, `g` general |
| `prompt` | The user prompt, verbatim. Non-empty, at most 8,000 characters (API limit). |
| `task_type` | `code` / `math` / `writing` / `general` |
| `tier` | `cheap` / `strong` |

No other fields. Difficulty is used to *decide* the tier (§3–4) but is not stored.

## 2. Dataset composition

- 100 prompts: 25 per task type.
- About half `cheap` and half `strong` overall, and within each task type.
- No personal data (names, emails, addresses, account numbers) — prompts are stored
  (TRD §7, Privacy).

## 3. Task type

Use the same definitions Jev is asked to apply (`app/core/classifier.py`,
`get_default_questions`): *classify the requested task into exactly one category based
on the work required.*

| Label | Definition |
|---|---|
| `code` | Programming, software development, code generation, debugging, or script execution |
| `math` | Mathematics, calculation, symbolic logic, proof, or formal problem solving |
| `writing` | Content generation, creative writing, copywriting, editing, or email correspondence |
| `general` | General conversation, facts, broad explanations, or queries that span multiple domains |

### Tie-break rules

1. **Label by the main deliverable.** Ask "what does the user want back?" A Python
   script that solves an equation → `code`. A proof that an algorithm terminates →
   `math`. A blog post about Kubernetes → `writing`.
2. **`general` is the fallback** when no single category dominates the work required.
3. If a prompt still feels genuinely ambiguous after rules 1–2, replace it with a
   clearer one rather than guessing; the eval measures routing, not label noise.

## 4. Difficulty (guidance only — not stored)

Question wording (`app/core/classifier.py`): *How difficult is it to produce a correct and
useful answer to the user's prompt?* Five levels, mapped to 1–5 (TRD §4):

| Level | Value | Meaning |
|---|---|---|
| trivial | 1 | Any model answers correctly in one step |
| easy | 2 | Routine; small chance of error |
| moderate | 3 | Some care needed, but a capable small model should manage |
| hard | 4 | Needs strong domain knowledge or careful multi-part work |
| expert | 5 | Specialist-level; small models are likely to fail |

## 5. Tier

Derived from the routing policy's HARD rule (TRD §5): strong if
`difficulty >= 3.5` or `p_reasoning >= 0.6`. On the five levels, 3.5 falls between
moderate and hard.

| Label | When |
|---|---|
| `strong` | Difficulty is **hard** or **expert**, **or** a good answer requires multiple dependent reasoning steps |
| `cheap` | Difficulty is **trivial**, **easy**, or **moderate**, **and** no multi-step reasoning is required |

"Multiple dependent reasoning steps" uses the Jev question's wording: *Does answering the
user's prompt well require multiple dependent reasoning steps?*

Never label the runtime outcomes `LOW_CONFIDENCE`, `JEV_ERROR`, `BUDGET`, or `CACHE`;
they are not properties of a prompt.

## 6. Examples

| id | Prompt | task_type | tier | Why |
|---|---|---|---|---|
| c-001 | Reverse a string in Python | code | cheap | trivial, single step |
| c-002 | Design a thread-safe distributed rate limiter that handles Redis failures | code | strong | expert, multi-step |
| m-001 | What is 2 + 2? | math | cheap | trivial |
| m-002 | Prove that there are infinitely many prime numbers | math | strong | proof needs dependent steps |
| w-001 | Write a polite email asking a professor for a two-day extension | writing | cheap | easy |
| g-001 | What is the capital of Japan? | general | cheap | trivial fact |
