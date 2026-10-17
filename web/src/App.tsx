import { useEffect, useState } from "react";

type ComponentStatus = "ok" | "error" | "not_configured";
type HealthReport = { status: "ok" | "degraded"; db: ComponentStatus; redis: ComponentStatus };
type HealthState = { kind: "loading" } | { kind: "report"; report: HealthReport } | { kind: "error"; message: string };

const COMPONENT_STATUSES: readonly unknown[] = ["ok", "error", "not_configured"];

// S2.14 decision U1: the response cache is shared by every client, so a cached answer can be returned to
// anyone who sends the same prompt. Shown on every page state, before any prompt can be entered.
const PERSONAL_DATA_NOTICE =
  "Do not enter personal or confidential information. Answers can be cached and returned to anyone who sends the same prompt.";

function isHealthReport(body: unknown): body is HealthReport {
  if (typeof body !== "object" || body === null) return false;
  const { status, db, redis } = body as Record<string, unknown>;
  return (status === "ok" || status === "degraded") && COMPONENT_STATUSES.includes(db) && COMPONENT_STATUSES.includes(redis);
}

function describe(name: string, status: ComponentStatus): string {
  const label = status === "ok" ? "ok" : status === "not_configured" ? "not configured" : "unavailable";
  return `${name} ${label}`;
}

// The API answers 200 when DB and Redis are healthy and 503 with the same report when degraded.
export default function App() {
  const [health, setHealth] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    async function loadHealth() {
      try {
        const response = await fetch("/api/health", { signal: controller.signal });
        if (!response.ok && response.status !== 503) {
          setHealth({ kind: "error", message: `API unavailable (HTTP ${response.status})` });
          return;
        }
        // A 503 may come from the platform rather than the API, with a non-JSON body (e.g. HTML).
        const body: unknown = response.ok ? await response.json() : await response.json().catch(() => undefined);
        if (isHealthReport(body)) {
          setHealth({ kind: "report", report: body });
        } else if (response.ok) {
          setHealth({ kind: "error", message: "API returned an unexpected health status" });
        } else {
          setHealth({ kind: "error", message: `API unavailable (HTTP ${response.status})` });
        }
      } catch (error: unknown) {
        if (controller.signal.aborted) return;
        const detail = error instanceof Error ? error.message : "unknown error";
        setHealth({ kind: "error", message: `API unreachable: ${detail}` });
      }
    }

    loadHealth();
    return () => controller.abort();
  }, []);

  const statusText =
    health.kind === "loading"
      ? "Checking API…"
      : health.kind === "report"
        ? `API ok · ${describe("DB", health.report.db)} · ${describe("Redis", health.report.redis)}`
        : health.message;

  return (
    <main>
      <h1>Switchyard</h1>
      <p role="note">{PERSONAL_DATA_NOTICE}</p>
      <p role="status">{statusText}</p>
    </main>
  );
}
