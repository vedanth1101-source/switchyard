import { useEffect, useState } from "react";

type HealthState = { kind: "loading" } | { kind: "ok" } | { kind: "error"; message: string };

// Task 0.3 shows API status only; DB and Redis indicators arrive with Task 0.5.
export default function App() {
  const [health, setHealth] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    async function loadHealth() {
      try {
        const response = await fetch("/api/health", { signal: controller.signal });
        if (!response.ok) {
          setHealth({ kind: "error", message: `API unavailable (HTTP ${response.status})` });
          return;
        }
        const body: { status?: unknown } = await response.json();
        setHealth(
          body.status === "ok"
            ? { kind: "ok" }
            : { kind: "error", message: "API returned an unexpected health status" },
        );
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
    health.kind === "loading" ? "Checking API…" : health.kind === "ok" ? "API ok" : health.message;

  return (
    <main>
      <h1>Switchyard</h1>
      <p role="status">{statusText}</p>
    </main>
  );
}
