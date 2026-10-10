# Project Flow: Switchyard

## 1. Request flow (`POST /api/route`)
```mermaid
sequenceDiagram
    participant B as Browser (React)
    participant A as FastAPI (Vercel fn)
    participant RL as Rate limiter (Redis)
    participant C as Cache (Redis)
    participant J as Jev
    participant P as Policy
    participant BG as Budget (Redis)
    participant E as Executor
    participant M as LLM provider
    participant D as Postgres

    B->>A: POST /api/route {prompt}
    A->>RL: incr ip:minute
    alt over limit
        A-->>B: 429
    end
    A->>C: get(key)
    alt cache hit
        C-->>A: decision + answer
    else miss
        A->>J: one call, 3 questions
        J-->>A: probabilities + confidences (or error)
        A->>P: decide_route(decision)
        P->>BG: strong cap ok?
        P-->>A: tier, model chain, reason
        A->>E: execute(chain, prompt, deadline 50 s)
        loop within deadline
            E->>M: chat completion
            M-->>E: answer or error → retry / next model
        end
        E-->>A: result + attempts
        A->>C: set(key, result, TTL 24h)
    end
    A->>D: insert request, decision, model calls
    A-->>B: RouteResponse
```

## 2. Routing decision
```mermaid
flowchart TD
    S[Prompt] --> RL{Under rate limit?}
    RL -- No --> E429[429 Too Many Requests]
    RL -- Yes --> CH{Cache hit?}
    CH -- Yes --> RC[Return cached<br/>reason: CACHE]
    CH -- No --> J{Jev OK?}
    J -- No --> SE[STRONG general<br/>JEV_ERROR]
    J -- Yes --> C{confidence ≥ threshold?}
    C -- No --> SL[STRONG general<br/>LOW_CONFIDENCE]
    C -- Yes --> H{difficulty ≥ 3.5 or<br/>p_reasoning ≥ 0.6?}
    H -- Yes --> SH[STRONG task model<br/>HARD]
    H -- No --> CE[CHEAP task model<br/>EASY]
    SE --> BG{Strong cap reached?}
    SL --> BG
    SH --> BG
    BG -- Yes --> CB[CHEAP<br/>BUDGET]
    BG -- No --> X[Execute]
    CE --> X
    CB --> X
```

## 3. Execution within the time budget
```mermaid
flowchart LR
    A[Model 1] -->|success| OK[Return]
    A -->|429/5xx/timeout and time left| R[Retry with backoff]
    R -->|success| OK
    R -->|still failing| B[Next model]
    A -->|other 4xx| B
    B -->|success| OK
    B -->|fails or deadline hit| F[502 + attempt log]
```

## 4. Evaluation (runs on laptop)
```mermaid
flowchart LR
    D[eval_prompts.jsonl] --> RUN[run_eval --dry-run]
    RUN --> PG[(Postgres: eval_runs, eval_items)]
    RUN --> MET[accuracy, confusion matrix,<br/>cheap share, est. savings, latency]
    MET --> SW[threshold sweep 0.4–0.9]
    SW --> REP[reports/*.md + *.json]
    PG --> UI[React Evals page]
    REP --> README[README results table]
```
Then run `--with-answers --limit 20` for the manual quality check.

## 5. Frontend flow
```mermaid
flowchart LR
    CHAT[Chat page] -->|POST /api/route| API
    DASH[Dashboard page] -->|GET /api/metrics| API
    EVALS[Evals page] -->|GET /api/evals| API
    CHAT --> DP[DecisionPanel: task, difficulty,<br/>confidence, tier, reason, attempts, cost]
```

## 6. Dev workflow per feature
Red test → Claude Code makes it green → refactor → commit referencing FR id
(e.g. `feat(policy): FR-3 low-confidence fallback`) → push → CI → Vercel preview deploy.
