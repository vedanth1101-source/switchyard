import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function respondWith(status: number, body: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({ ok: status >= 200 && status < 300, status, json: async () => body });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const PERSONAL_DATA_NOTICE =
  "Do not enter personal or confidential information. Answers can be cached and returned to anyone who sends the same prompt.";

describe("App", () => {
  it("shows the personal-data notice before the health check finishes", () => {
    respondWith(200, { status: "ok", db: "ok", redis: "ok" });

    render(<App />);

    expect(screen.getByRole("note").textContent).toBe(PERSONAL_DATA_NOTICE);
  });

  it("keeps the personal-data notice when the API is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    render(<App />);

    expect(await screen.findByText("API unreachable: Failed to fetch")).toBeTruthy();
    expect(screen.getByRole("note").textContent).toBe(PERSONAL_DATA_NOTICE);
  });

  it("renders health status", async () => {
    const fetchMock = respondWith(200, { status: "ok", db: "ok", redis: "ok" });

    render(<App />);

    expect(screen.getByText("Checking API…")).toBeTruthy();
    expect(await screen.findByText("API ok · DB ok · Redis ok")).toBeTruthy();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/health");
  });

  it("identifies an unavailable database when the API reports degraded", async () => {
    respondWith(503, { status: "degraded", db: "error", redis: "ok" });

    render(<App />);

    expect(await screen.findByText("API ok · DB unavailable · Redis ok")).toBeTruthy();
  });

  it("identifies unconfigured and unavailable dependencies together", async () => {
    respondWith(503, { status: "degraded", db: "not_configured", redis: "error" });

    render(<App />);

    expect(await screen.findByText("API ok · DB not configured · Redis unavailable")).toBeTruthy();
  });

  it("renders an error when the health request fails", async () => {
    respondWith(502, {});

    render(<App />);

    expect(await screen.findByText("API unavailable (HTTP 502)")).toBeTruthy();
  });

  it("renders an error when a 503 body is not a health report", async () => {
    respondWith(503, { detail: "Service Unavailable" });

    render(<App />);

    expect(await screen.findByText("API unavailable (HTTP 503)")).toBeTruthy();
  });

  it("renders HTTP 503 when the 503 body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => {
          throw new SyntaxError("Unexpected token '<', \"<!DOCTYPE \"... is not valid JSON");
        },
      }),
    );

    render(<App />);

    expect(await screen.findByText("API unavailable (HTTP 503)")).toBeTruthy();
    expect(screen.queryByText(/API unreachable/)).toBeNull();
  });

  it("renders an error when the API is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    render(<App />);

    expect(await screen.findByText("API unreachable: Failed to fetch")).toBeTruthy();
  });
});
