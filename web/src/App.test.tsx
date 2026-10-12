import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders health status", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(screen.getByText("Checking API…")).toBeTruthy();
    expect(await screen.findByText("API ok")).toBeTruthy();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/health");
  });

  it("renders an error when the health request fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 502, json: async () => ({}) }));

    render(<App />);

    expect(await screen.findByText("API unavailable (HTTP 502)")).toBeTruthy();
  });
});
