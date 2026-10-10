# Jev Playground Spike — Initial Observations

**Project:** Switchyard  
**Purpose:** Validate whether Jev can provide structured signals for model routing.  
**Status:** Initial manual exploration; not a formal benchmark.

## Questions configured in Playground

### Question 1 — Task type
Prompt: `Classify the user's requested task into exactly one category based on the work required.`

Choice labels:
- `code`
- `math`
- `writing`
- `general`

### Question 2 — Difficulty
Prompt: `How difficult is it to produce a correct and useful answer to the user's prompt?`

Choice labels:
1. `trivial`
2. `easy`
3. `moderate`
4. `hard`
5. `expert`

Note: the Playground screenshots show this question configured as **Choice**, not **Score**. The API/SDK schema must be verified before deciding how to represent the result in code.

### Question 3 — Multi-step reasoning
Prompt: `Does answering the user's prompt well require multiple dependent reasoning steps?`

Type: Boolean. The Playground displays probabilities for Yes and No.

## Observed experiments

### Experiment A — Simple arithmetic
State: `What is 2 + 2?`

Observed:
- Difficulty distribution: trivial 97%, easy 2%, moderate 1%, hard 0%, expert 0%.
- Reasoning: Yes 17%, No 83%.
- Latency displayed: 871 ms (server + network).

Task classification output was not visible in the saved screenshot excerpt.

### Experiment B — Simple coding
State: `Write a Python function that reverses a string.`

Observed:
- Task type: `code` — 100%.
- Difficulty: `trivial` — 99%, `easy` — 1%, remaining levels 0%.
- Reasoning: Yes 19%, No 81%.

### Experiment C — Complex distributed systems coding
State: `Design a thread-safe rate limiter for a distributed API serving millions of requests. Explain how to handle concurrent requests across multiple servers, enforce limits atomically, handle Redis failures, and test correctness under race conditions.`

Observed:
- Difficulty: `expert` — 61%, `hard` — 38%, `moderate` — 1%, trivial/easy 0%.
- Reasoning: Yes 96%, No 4%.
- Task classification was expected to be code; retain the raw Playground output if the exact task distribution is needed.

### Experiment D — Writing
State: `Write a friendly, professional email asking my professor for a two-day extension on an assignment because I have multiple deadlines this week.`

Observed:
- Task type: `writing` — 100%.
- Difficulty: `easy` — 51%, `trivial` — 49%, moderate/hard/expert 0%.
- Reasoning: Yes 48%, No 52%.
- Latency displayed: 545 ms (server + network).

## Initial interpretation

These few observations are promising:
- The simple and complex coding prompts receive very different difficulty distributions.
- The reasoning probability is much higher for the complex distributed-systems prompt than for the simple coding prompt.
- The email prompt is classified as writing.

These examples do **not** establish general accuracy, calibration, repeatability, or production latency. Continue with a labeled dataset, repeatability tests, and API-level verification.

## Next validation steps
1. Run a math proof prompt: `Prove that there are infinitely many prime numbers.`
2. Save the exact Playground outputs for all three questions.
3. Verify the official `typesafe-sdk` package, authentication, `system_one()` method, question types, response schema, confidence fields, and timeout/retry behavior.
4. Make one real Python SDK call and save a sanitized response fixture.
5. Record access path, usage pricing, and observed latency before finalizing the $0 constraint.
