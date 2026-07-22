export class SessionRegistry {
  #active = null;

  activate(session) {
    this.#active = session;
  }

  owns(session) {
    return this.#active === session;
  }

  hasActive() {
    return this.#active !== null;
  }

  current() {
    return this.#active;
  }

  retire(session) {
    if (!this.owns(session)) {
      return false;
    }
    this.#active = null;
    return true;
  }

  take() {
    const session = this.#active;
    this.#active = null;
    return session;
  }
}

export function buildIceServers(credentials) {
  const iceServers = [{ urls: "stun:stun.l.google.com:19302" }];
  if (!credentials.turnUrl) {
    return iceServers;
  }

  let turnUrl;
  try {
    turnUrl = new URL(credentials.turnUrl);
  } catch {
    throw new Error("Enter a valid TURN URL.");
  }
  if (!["turn:", "turns:"].includes(turnUrl.protocol)) {
    throw new Error("The relay URL must use turn: or turns:.");
  }

  iceServers.push({
    urls: credentials.turnUrl,
    username: credentials.turnUsername,
    credential: credentials.turnCredential,
  });
  return iceServers;
}
