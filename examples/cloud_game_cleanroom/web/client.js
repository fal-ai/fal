import { buildIceServers, SessionRegistry } from "./session.js";

const elements = {
  form: document.querySelector("#connectionForm"),
  wmaUrl: document.querySelector("#wmaUrl"),
  appId: document.querySelector("#appId"),
  falKey: document.querySelector("#falKey"),
  turnUrl: document.querySelector("#turnUrl"),
  turnUsername: document.querySelector("#turnUsername"),
  turnCredential: document.querySelector("#turnCredential"),
  formError: document.querySelector("#formError"),
  connectButton: document.querySelector("#connectButton"),
  disconnectButton: document.querySelector("#disconnectButton"),
  connectionLabel: document.querySelector("#connectionLabel"),
  sessionState: document.querySelector(".session-state"),
  stage: document.querySelector("#stage"),
  stageEmpty: document.querySelector("#stageEmpty"),
  stageLoading: document.querySelector("#stageLoading"),
  stageError: document.querySelector("#stageError"),
  stageErrorMessage: document.querySelector("#stageErrorMessage"),
  stream: document.querySelector("#stream"),
  hud: document.querySelector("#hud"),
  focusButton: document.querySelector("#focusButton"),
  audioButton: document.querySelector("#audioButton"),
  scoreLabel: document.querySelector("#scoreLabel"),
  livesLabel: document.querySelector("#livesLabel"),
  levelLabel: document.querySelector("#levelLabel"),
  score: document.querySelector("#score"),
  lives: document.querySelector("#lives"),
  level: document.querySelector("#level"),
  rtt: document.querySelector("#rtt"),
  fps: document.querySelector("#fps"),
  jitter: document.querySelector("#jitter"),
  loss: document.querySelector("#loss"),
};

const controls = {
  keys: new Set(),
  mouseButtons: new Set(),
  mouseDx: 0,
  mouseDy: 0,
  sequence: 0,
  lastGamepad: "null",
  heartbeatTick: 0,
};

const sessions = new SessionRegistry();
elements.stream.muted = true;

function setConnectionState(state, label) {
  elements.sessionState.dataset.state = state;
  elements.connectionLabel.textContent = label;
}

function setStageState(state, message = "") {
  elements.stageEmpty.hidden = state !== "idle";
  elements.stageLoading.hidden = state !== "loading";
  elements.stageError.hidden = state !== "error";
  elements.hud.hidden = state !== "playing";
  elements.focusButton.hidden = state !== "playing";
  elements.audioButton.hidden = state !== "playing" || !elements.stream.muted;
  if (state === "error") {
    elements.stageErrorMessage.textContent = message;
  }
}

function setFormBusy(busy) {
  elements.connectButton.disabled = busy;
  elements.connectButton.textContent = busy ? "Connecting" : "Start session";
  elements.disconnectButton.disabled = !busy;
  elements.wmaUrl.disabled = busy;
  elements.appId.disabled = busy;
  elements.falKey.disabled = busy;
  elements.turnUrl.disabled = busy;
  elements.turnUsername.disabled = busy;
  elements.turnCredential.disabled = busy;
}

async function waitForIceGathering(connection, timeoutMs = 7000) {
  if (connection.iceGatheringState === "complete") {
    return;
  }

  await new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) {
        return;
      }
      settled = true;
      window.clearTimeout(timeout);
      connection.removeEventListener("icegatheringstatechange", onStateChange);
      resolve();
    };
    const onStateChange = () => {
      if (connection.iceGatheringState === "complete") {
        finish();
      }
    };
    const timeout = window.setTimeout(finish, timeoutMs);
    connection.addEventListener("icegatheringstatechange", onStateChange);
  });
}

function readGamepad() {
  const pad = navigator.getGamepads?.().find(Boolean);
  if (!pad) {
    return null;
  }
  return {
    axes: Array.from(pad.axes.slice(0, 4), (value) =>
      Number(value.toFixed(3)),
    ),
    buttons: Array.from(pad.buttons, (button) => button.pressed),
  };
}

function isActive(session) {
  return sessions.owns(session);
}

function assertActive(session) {
  if (!isActive(session)) {
    throw new DOMException("Session attempt was superseded", "AbortError");
  }
}

function sendInput(session, gamepad = readGamepad()) {
  if (!isActive(session) || session.inputChannel?.readyState !== "open") {
    return;
  }
  const mouse = {
    dx: Math.round(controls.mouseDx),
    dy: Math.round(controls.mouseDy),
    buttons: Array.from(controls.mouseButtons),
  };
  controls.mouseDx = 0;
  controls.mouseDy = 0;
  session.inputChannel.send(
    JSON.stringify({
      type: "input",
      seq: controls.sequence++,
      keys: Array.from(controls.keys),
      gamepad,
      mouse,
    }),
  );
}

function sendNeutralInput(session) {
  if (!isActive(session) || session.inputChannel?.readyState !== "open") {
    return;
  }
  session.inputChannel.send(
    JSON.stringify({
      type: "input",
      seq: controls.sequence++,
      keys: [],
      gamepad: null,
      mouse: { dx: 0, dy: 0, buttons: [] },
    }),
  );
}

function startInputLoop(session) {
  window.clearInterval(session.inputTimer);
  session.inputTimer = window.setInterval(() => {
    const gamepad = readGamepad();
    const serialized = JSON.stringify(gamepad);
    const gamepadChanged = serialized !== controls.lastGamepad;
    controls.lastGamepad = serialized;
    controls.heartbeatTick = (controls.heartbeatTick + 1) % 6;
    if (
      gamepadChanged ||
      controls.mouseDx !== 0 ||
      controls.mouseDy !== 0 ||
      controls.heartbeatTick === 0
    ) {
      sendInput(session, gamepad);
    }
  }, 1000 / 60);
}

function handleServerMessage(session, event) {
  if (!isActive(session)) {
    return;
  }
  let message;
  try {
    message = JSON.parse(event.data);
  } catch {
    return;
  }

  if (message.type === "game_state") {
    if (message.backend === "freedoom") {
      elements.scoreLabel.textContent = "Game";
      elements.livesLabel.textContent = "State";
      elements.levelLabel.textContent = "Controls";
      elements.score.textContent = "FREEDOOM";
      elements.lives.textContent = String(message.phase ?? "starting").toUpperCase();
      elements.level.textContent = "MOUSE + KEYS";
    } else {
      elements.scoreLabel.textContent = "Score";
      elements.livesLabel.textContent = "Lives";
      elements.levelLabel.textContent = "Level";
      elements.score.textContent = String(message.score ?? 0).padStart(6, "0");
      elements.lives.textContent = String(message.lives ?? 0);
      elements.level.textContent = String(message.level ?? 1);
    }
  }
}

function attachInputChannel(session, channel) {
  if (channel.label !== "input") {
    return;
  }
  session.inputChannel = channel;
  channel.addEventListener("open", () => startInputLoop(session));
  channel.addEventListener("message", (event) =>
    handleServerMessage(session, event),
  );
  channel.addEventListener("close", () => {
    window.clearInterval(session.inputTimer);
  });
}

async function collectStats(session) {
  if (!isActive(session)) {
    return;
  }
  const stats = await session.peer.getStats();
  let rtt = null;
  let fps = null;
  let jitter = null;
  let packetsLost = null;

  stats.forEach((report) => {
    if (
      report.type === "candidate-pair" &&
      report.state === "succeeded" &&
      report.currentRoundTripTime !== undefined
    ) {
      rtt = report.currentRoundTripTime * 1000;
    }
    if (
      report.type === "inbound-rtp" &&
      (report.kind ?? report.mediaType) === "video"
    ) {
      fps = report.framesPerSecond;
      jitter = report.jitter * 1000;
      packetsLost = report.packetsLost;
    }
  });

  elements.rtt.textContent = rtt === null ? "—" : `${Math.round(rtt)} ms`;
  elements.fps.textContent = fps == null ? "—" : `${Math.round(fps)} fps`;
  elements.jitter.textContent =
    jitter === null ? "—" : `${jitter.toFixed(1)} ms`;
  elements.loss.textContent =
    packetsLost === null ? "—" : String(packetsLost);
}

function startStatsLoop(session) {
  window.clearInterval(session.statsTimer);
  session.statsTimer = window.setInterval(() => {
    collectStats(session).catch(() => {});
  }, 1000);
}

function authHeaders(falKey) {
  return {
    Authorization: `Key ${falKey}`,
    "Content-Type": "application/json",
  };
}

async function heartbeat(session) {
  if (!isActive(session) || !session.sessionId) {
    return;
  }
  const response = await fetch(
    `${session.credentials.wmaUrl}/session/heartbeat`,
    {
      method: "POST",
      headers: authHeaders(session.credentials.falKey),
      body: JSON.stringify({ session_id: session.sessionId }),
      signal: session.abortController.signal,
    },
  );
  if (!response.ok) {
    throw new Error(`Heartbeat failed with HTTP ${response.status}`);
  }
  const result = await response.json();
  if (!result.alive) {
    throw new Error("WMA session is no longer alive");
  }
}

function startHeartbeatLoop(session) {
  window.clearInterval(session.heartbeatTimer);
  session.heartbeatTimer = window.setInterval(() => {
    heartbeat(session).catch((error) => failSession(session, error));
  }, 5000);
}

async function connect(credentials) {
  disconnect();
  setFormBusy(true);
  setConnectionState("connecting", "Negotiating session");
  setStageState("loading");
  elements.formError.textContent = "";

  let connection;
  try {
    connection = new RTCPeerConnection({
      iceServers: buildIceServers(credentials),
    });
  } catch (error) {
    showConnectionError(error);
    return;
  }
  const session = {
    peer: connection,
    inputChannel: null,
    sessionId: null,
    credentials,
    abortController: new AbortController(),
    heartbeatTimer: null,
    inputTimer: null,
    statsTimer: null,
    disconnectTimer: null,
    remoteStream: new MediaStream(),
  };
  sessions.activate(session);

  try {
    connection.addTransceiver("video", { direction: "recvonly" });
    connection.addTransceiver("audio", { direction: "recvonly" });
    attachInputChannel(
      session,
      connection.createDataChannel("input", {
        ordered: false,
        maxRetransmits: 0,
      }),
    );
  } catch (error) {
    failSession(session, error);
    return;
  }

  connection.addEventListener("track", (event) => {
    if (!isActive(session)) {
      return;
    }
    session.remoteStream.addTrack(event.track);
    elements.stream.srcObject = session.remoteStream;
    elements.stream.play().catch(() => {});
  });
  connection.addEventListener("connectionstatechange", () => {
    if (!isActive(session)) {
      return;
    }
    if (connection.connectionState === "connected") {
      window.clearTimeout(session.disconnectTimer);
      session.disconnectTimer = null;
      setConnectionState("connected", "Live from fal runner");
      setStageState("playing");
      elements.stage.focus();
      startStatsLoop(session);
    } else if (
      connection.connectionState === "disconnected" &&
      session.disconnectTimer === null
    ) {
      setConnectionState("connecting", "Reconnecting");
      session.disconnectTimer = window.setTimeout(() => {
        failSession(session, new Error("WebRTC did not recover"));
      }, 5000);
    } else if (["failed", "closed"].includes(connection.connectionState)) {
      failSession(
        session,
        new Error(`WebRTC state: ${connection.connectionState}`),
      );
    }
  });

  try {
    const offer = await connection.createOffer();
    await connection.setLocalDescription(offer);
    await waitForIceGathering(connection);
    assertActive(session);

    const response = await fetch(`${credentials.wmaUrl}/session`, {
      method: "POST",
      headers: authHeaders(credentials.falKey),
      signal: session.abortController.signal,
      body: JSON.stringify({
        app_id: credentials.appId,
        sdp: connection.localDescription.sdp,
        type: connection.localDescription.type,
      }),
    });
    const body = await response.text();
    assertActive(session);
    if (!response.ok) {
      throw new Error(
        `WMA returned HTTP ${response.status}: ${body.slice(0, 240)}`,
      );
    }

    const answer = JSON.parse(body);
    session.sessionId = answer.session_id;
    await connection.setRemoteDescription(
      new RTCSessionDescription({
        sdp: answer.sdp,
        type: answer.type,
      }),
    );
    assertActive(session);
    startHeartbeatLoop(session);
  } catch (error) {
    if (error.name !== "AbortError" && isActive(session)) {
      failSession(session, error);
    }
  }
}

function resetTelemetry() {
  elements.rtt.textContent = "—";
  elements.fps.textContent = "—";
  elements.jitter.textContent = "—";
  elements.loss.textContent = "—";
}

function closeSession(session) {
  session.abortController.abort();
  window.clearInterval(session.heartbeatTimer);
  window.clearInterval(session.inputTimer);
  window.clearInterval(session.statsTimer);
  window.clearTimeout(session.disconnectTimer);
  controls.keys.clear();
  controls.mouseButtons.clear();
  controls.mouseDx = 0;
  controls.mouseDy = 0;
  controls.lastGamepad = "null";
  controls.heartbeatTick = 0;
  session.inputChannel?.close();
  session.peer.close();
}

function disconnect() {
  const session = sessions.take();
  if (session) {
    closeSession(session);
  }
  controls.keys.clear();
  if (document.pointerLockElement === elements.stage) {
    document.exitPointerLock();
  }
  elements.stream.srcObject = null;
  elements.stream.muted = true;
  setFormBusy(false);
  setConnectionState("idle", "Session offline");
  setStageState("idle");
  resetTelemetry();
}

function failSession(session, error) {
  if (!sessions.retire(session)) {
    return;
  }
  closeSession(session);
  showConnectionError(error);
}

function showConnectionError(error) {
  const message = error instanceof Error ? error.message : String(error);
  controls.keys.clear();
  controls.mouseButtons.clear();
  controls.mouseDx = 0;
  controls.mouseDy = 0;
  elements.stream.srcObject = null;
  setFormBusy(false);
  setConnectionState("error", "Session failed");
  setStageState("error", message);
  elements.formError.textContent = message;
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const credentials = {
    wmaUrl: elements.wmaUrl.value.trim().replace(/\/$/, ""),
    appId: elements.appId.value.trim(),
    falKey: elements.falKey.value.trim(),
    turnUrl: elements.turnUrl.value.trim(),
    turnUsername: elements.turnUsername.value.trim(),
    turnCredential: elements.turnCredential.value,
  };
  if (!credentials.wmaUrl || !credentials.appId || !credentials.falKey) {
    elements.formError.textContent = "Complete all three connection fields.";
    return;
  }
  let endpoint;
  try {
    endpoint = new URL(credentials.wmaUrl);
  } catch {
    elements.formError.textContent = "Enter a valid WMA endpoint URL.";
    return;
  }
  const isLocal =
    endpoint.hostname === "localhost" || endpoint.hostname === "127.0.0.1";
  if (
    endpoint.protocol !== "https:" &&
    !(isLocal && endpoint.protocol === "http:")
  ) {
    elements.formError.textContent =
      "The WMA endpoint must use HTTPS, except on localhost.";
    return;
  }
  const isFalHost =
    endpoint.hostname === "fal.run" || endpoint.hostname.endsWith(".fal.run");
  if (
    !isFalHost &&
    !isLocal &&
    !window.confirm(
      `Send your fal key to ${endpoint.hostname}? Only continue if you trust this endpoint.`,
    )
  ) {
    return;
  }
  try {
    buildIceServers(credentials);
  } catch (error) {
    elements.formError.textContent = error.message;
    return;
  }
  connect(credentials);
});

elements.disconnectButton.addEventListener("click", disconnect);
elements.focusButton.addEventListener("click", async () => {
  elements.stage.focus();
  try {
    await elements.stage.requestPointerLock({ unadjustedMovement: true });
  } catch {
    await elements.stage.requestPointerLock();
  }
});
elements.audioButton.addEventListener("click", async () => {
  elements.stream.muted = false;
  await elements.stream.play();
  elements.audioButton.hidden = true;
});

window.addEventListener("keydown", (event) => {
  if (!sessions.hasActive() || document.activeElement !== elements.stage) {
    return;
  }
  if (
    [
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
      "Space",
      "KeyW",
      "KeyA",
      "KeyS",
      "KeyD",
      "ControlLeft",
      "ControlRight",
      "ShiftLeft",
      "ShiftRight",
      "Escape",
      "Digit1",
      "Digit2",
      "Digit3",
      "Digit4",
      "Digit5",
      "Digit6",
      "Digit7",
    ].includes(event.code)
  ) {
    event.preventDefault();
    const size = controls.keys.size;
    controls.keys.add(event.code);
    if (controls.keys.size !== size) {
      const session = sessions.current();
      if (session) {
        sendInput(session);
      }
    }
  }
});

window.addEventListener("keyup", (event) => {
  if (controls.keys.delete(event.code)) {
    const session = sessions.current();
    if (session) {
      sendInput(session);
    }
  }
});

elements.stage.addEventListener("mousemove", (event) => {
  if (document.pointerLockElement !== elements.stage) {
    return;
  }
  controls.mouseDx += event.movementX;
  controls.mouseDy += event.movementY;
});

elements.stage.addEventListener("mousedown", (event) => {
  if (document.pointerLockElement !== elements.stage) {
    return;
  }
  event.preventDefault();
  controls.mouseButtons.add(event.button + 1);
  const session = sessions.current();
  if (session) {
    sendInput(session);
  }
});

window.addEventListener("mouseup", (event) => {
  if (controls.mouseButtons.delete(event.button + 1)) {
    const session = sessions.current();
    if (session) {
      sendInput(session);
    }
  }
});

function releaseControls() {
  controls.keys.clear();
  controls.mouseButtons.clear();
  controls.mouseDx = 0;
  controls.mouseDy = 0;
  const session = sessions.current();
  if (session) {
    sendNeutralInput(session);
  }
}

window.addEventListener("blur", releaseControls);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    releaseControls();
  }
});
document.addEventListener("pointerlockchange", () => {
  if (document.pointerLockElement !== elements.stage) {
    controls.mouseButtons.clear();
    const session = sessions.current();
    if (session) {
      sendInput(session);
    }
  }
});
window.addEventListener("beforeunload", disconnect);
