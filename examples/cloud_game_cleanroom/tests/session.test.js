import { describe, expect, test } from "bun:test";

import { buildIceServers, SessionRegistry } from "../web/session.js";

describe("SessionRegistry", () => {
  test("rejects stale failures after a session is superseded", () => {
    const registry = new SessionRegistry();
    const first = { name: "first" };
    const second = { name: "second" };

    registry.activate(first);
    registry.activate(second);

    expect(registry.hasActive()).toBe(true);
    expect(registry.owns(first)).toBe(false);
    expect(registry.retire(first)).toBe(false);
    expect(registry.owns(second)).toBe(true);
  });

  test("take invalidates the active attempt before cleanup", () => {
    const registry = new SessionRegistry();
    const session = { name: "active" };
    registry.activate(session);

    expect(registry.current()).toBe(session);
    expect(registry.take()).toBe(session);
    expect(registry.current()).toBe(null);
    expect(registry.hasActive()).toBe(false);
    expect(registry.owns(session)).toBe(false);
  });
});

describe("buildIceServers", () => {
  test("adds a configured TURN relay", () => {
    expect(
      buildIceServers({
        turnUrl: "turns:relay.example.com:5349",
        turnUsername: "player",
        turnCredential: "short-lived-secret",
      }),
    ).toEqual([
      { urls: "stun:stun.l.google.com:19302" },
      {
        urls: "turns:relay.example.com:5349",
        username: "player",
        credential: "short-lived-secret",
      },
    ]);
  });

  test("rejects malformed and non-TURN relay URLs", () => {
    expect(() =>
      buildIceServers({
        turnUrl: "not a URL",
        turnUsername: "",
        turnCredential: "",
      }),
    ).toThrow("Enter a valid TURN URL.");
    expect(() =>
      buildIceServers({
        turnUrl: "https://relay.example.com",
        turnUsername: "",
        turnCredential: "",
      }),
    ).toThrow("The relay URL must use turn: or turns:.");
  });
});
