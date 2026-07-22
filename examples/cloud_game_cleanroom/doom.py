from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from engine import HEIGHT, WIDTH, OrbitBreaker

FPS = 30
NATIVE_FPS = 60
DISPLAY = ":99"
PULSE_SINK = "fal_game"
PULSE_MONITOR = f"{PULSE_SINK}.monitor"
FREEDOOM_WAD = Path("/usr/share/games/doom/freedoom1.wad")
FRAME_BYTES = WIDTH * HEIGHT * 3
CRISPY_DOOM_PATHS = (
    Path("/opt/crispy-doom/bin/crispy-doom"),
    Path("/usr/local/games/crispy-doom"),
    Path("/usr/local/bin/crispy-doom"),
    Path("/usr/games/crispy-doom"),
)

KEY_MAP = {
    "ArrowUp": "Up",
    "ArrowDown": "Down",
    "ArrowLeft": "Left",
    "ArrowRight": "Right",
    "KeyW": "Up",
    "KeyS": "Down",
    "KeyA": "comma",
    "KeyD": "period",
    "Space": "space",
    "ControlLeft": "Control_L",
    "ControlRight": "Control_R",
    "ShiftLeft": "Shift_L",
    "ShiftRight": "Shift_R",
    "Escape": "Escape",
    "Digit1": "1",
    "Digit2": "2",
    "Digit3": "3",
    "Digit4": "4",
    "Digit5": "5",
    "Digit6": "6",
    "Digit7": "7",
}


class OrbitBackend:
    mode = "orbit"
    width = WIDTH
    height = HEIGHT
    fps = FPS

    def __init__(self) -> None:
        self._game = OrbitBreaker()
        self._last_frame_at = time.monotonic()

    def apply_input(self, message: dict[str, Any]) -> None:
        self._game.apply_input(message)

    def restart(self) -> None:
        self._game.reset()

    def read_rgb(self) -> bytes:
        now = time.monotonic()
        self._game.advance(now - self._last_frame_at)
        self._last_frame_at = now
        return self._game.render_rgb()

    def snapshot(self) -> dict[str, int | str]:
        return {"backend": self.mode, **self._game.snapshot()}

    def close(self) -> None:
        pass


class XTestInput:
    def __init__(self, display_name: str) -> None:
        self._x11 = ctypes.CDLL("libX11.so.6")
        self._xtst = ctypes.CDLL("libXtst.so.6")
        self._lock = threading.RLock()
        self._configure_signatures()
        if not self._x11.XInitThreads():
            raise RuntimeError("Xlib thread initialization failed")
        self._display = self._x11.XOpenDisplay(display_name.encode())
        if not self._display:
            raise RuntimeError(f"Could not open X display {display_name}")
        self._pressed: set[str] = set()
        self._pressed_buttons: set[int] = set()

    def _configure_signatures(self) -> None:
        self._x11.XInitThreads.restype = ctypes.c_int
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
        self._x11.XStringToKeysym.restype = ctypes.c_ulong
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XKeysymToKeycode.restype = ctypes.c_uint
        self._x11.XFlush.argtypes = [ctypes.c_void_p]
        self._xtst.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self._xtst.XTestFakeKeyEvent.restype = ctypes.c_int
        self._xtst.XTestFakeButtonEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self._xtst.XTestFakeButtonEvent.restype = ctypes.c_int
        self._xtst.XTestFakeRelativeMotionEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self._xtst.XTestFakeRelativeMotionEvent.restype = ctypes.c_int

    def update(self, keysyms: set[str]) -> None:
        with self._lock:
            if not self._display:
                return
            for keysym in self._pressed - keysyms:
                self._send(keysym, pressed=False)
            for keysym in keysyms - self._pressed:
                self._send(keysym, pressed=True)
            self._pressed = set(keysyms)
            self._x11.XFlush(self._display)

    def mouse(self, dx: int, dy: int, buttons: set[int]) -> None:
        with self._lock:
            if not self._display:
                return
            if dx or dy:
                if not self._xtst.XTestFakeRelativeMotionEvent(
                    self._display, dx, dy, 0
                ):
                    raise RuntimeError("XTest rejected relative mouse motion")
            for button in self._pressed_buttons - buttons:
                self._send_button(button, pressed=False)
            for button in buttons - self._pressed_buttons:
                self._send_button(button, pressed=True)
            self._pressed_buttons = set(buttons)
            self._x11.XFlush(self._display)

    def _send_button(self, button: int, *, pressed: bool) -> None:
        if not self._xtst.XTestFakeButtonEvent(
            self._display, button, int(pressed), 0
        ):
            raise RuntimeError(f"XTest rejected mouse button: {button}")

    def _send(self, keysym_name: str, *, pressed: bool) -> None:
        keysym = self._x11.XStringToKeysym(keysym_name.encode())
        keycode = self._x11.XKeysymToKeycode(self._display, keysym)
        if not keycode:
            raise RuntimeError(f"Unknown X11 keysym: {keysym_name}")
        if not self._xtst.XTestFakeKeyEvent(self._display, keycode, int(pressed), 0):
            raise RuntimeError(f"XTest rejected keysym: {keysym_name}")

    def close(self) -> None:
        with self._lock:
            if not self._display:
                return
            self.update(set())
            self.mouse(0, 0, set())
            self._x11.XCloseDisplay(self._display)
            self._display = None


class DoomSession:
    mode = "freedoom"
    width = WIDTH
    height = HEIGHT
    fps = FPS

    def __init__(
        self,
        *,
        display_name: str = DISPLAY,
        executable: str | None = None,
        wad_path: Path = FREEDOOM_WAD,
    ) -> None:
        self.display_name = display_name
        self.executable = executable or _find_crispy_doom()
        self.wad_path = wad_path
        self._workspace: tempfile.TemporaryDirectory[str] | None = None
        self._xvfb: subprocess.Popen[bytes] | None = None
        self._game: subprocess.Popen[bytes] | None = None
        self._capture: subprocess.Popen[bytes] | None = None
        self._audio: subprocess.Popen[bytes] | None = None
        self.pulse_server: str | None = None
        self._input: XTestInput | None = None
        self._started_at = 0.0
        self._last_sequence = -1

    @classmethod
    def is_available(cls) -> bool:
        return not cls.availability_issues()

    @classmethod
    def availability_issues(cls) -> list[str]:
        issues = []
        if _find_crispy_doom() is None:
            issues.append("missing executable: crispy-doom")
        issues.extend(
            f"missing executable: {command}"
            for command in ("Xvfb", "ffmpeg", "xdotool")
            if shutil.which(command) is None
        )
        if not FREEDOOM_WAD.is_file():
            issues.append(f"missing game data: {FREEDOOM_WAD}")
        return issues

    @classmethod
    def require_available(cls) -> None:
        issues = cls.availability_issues()
        if issues:
            raise RuntimeError("Freedoom runtime is unavailable: " + "; ".join(issues))

    def start(self, *, capture: bool = True, audio: bool = False) -> None:
        if not self.executable:
            raise RuntimeError("crispy-doom is not installed")
        if not self.wad_path.is_file():
            raise RuntimeError(f"Freedoom WAD not found: {self.wad_path}")

        try:
            self._workspace = tempfile.TemporaryDirectory(prefix="fal-freedoom-")
            workspace = Path(self._workspace.name)
            environment = {
                **os.environ,
                "DISPLAY": self.display_name,
                "HOME": str(workspace),
                "SDL_VIDEODRIVER": "x11",
            }
            if audio:
                if os.geteuid() == 0:
                    workspace.chmod(0o711)
                runtime_dir = _prepare_audio_runtime(workspace)
                environment["XDG_RUNTIME_DIR"] = str(runtime_dir)
                self._start_audio(environment)
                environment["SDL_AUDIODRIVER"] = "pulseaudio"
            else:
                environment["SDL_AUDIODRIVER"] = "dummy"
            self._xvfb = self._spawn(
                [
                    "Xvfb",
                    self.display_name,
                    "-screen",
                    "0",
                    f"{WIDTH}x{HEIGHT}x24",
                    "-nolisten",
                    "tcp",
                    "-noreset",
                    "-ac",
                ],
                environment,
            )
            self._input = self._wait_for_display()
            self._game = self._spawn(
                [
                    self.executable,
                    "-iwad",
                    str(self.wad_path),
                    "-warp",
                    "1",
                    "1",
                    "-skill",
                    "3",
                    "-window",
                    "-geometry",
                    f"{WIDTH}x{HEIGHT}",
                    "-config",
                    str(workspace / "default.cfg"),
                    "-extraconfig",
                    str(workspace / "crispy-doom.cfg"),
                ],
                environment,
            )
            self._wait_for_process(self._game, "Crispy Doom")
            subprocess.run(
                [
                    "xdotool",
                    "search",
                    "--sync",
                    "--onlyvisible",
                    "--limit",
                    "1",
                    "--name",
                    "Crispy Doom",
                    "windowfocus",
                ],
                check=True,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            if capture:
                self._capture = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-f",
                        "x11grab",
                        "-draw_mouse",
                        "0",
                        "-framerate",
                        str(FPS),
                        "-video_size",
                        f"{WIDTH}x{HEIGHT}",
                        "-i",
                        f"{self.display_name}.0+0,0",
                        "-pix_fmt",
                        "rgb24",
                        "-f",
                        "rawvideo",
                        "pipe:1",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    start_new_session=True,
                )
                self._wait_for_process(self._capture, "FFmpeg capture")
            self._started_at = time.monotonic()
        except BaseException:
            self.close()
            raise

    def _start_audio(self, environment: dict[str, str]) -> None:
        socket_path = Path(environment["XDG_RUNTIME_DIR"]) / "pulse.sock"
        log_path = Path(environment["XDG_RUNTIME_DIR"]) / "pulseaudio.log"
        self.pulse_server = f"unix:{socket_path}"
        environment["PULSE_SERVER"] = self.pulse_server
        command = [
            "pulseaudio",
            "--daemonize=no",
            "--exit-idle-time=-1",
            "--disallow-exit",
            "--disable-shm=true",
            "--realtime=no",
            "--high-priority=no",
            "--use-pid-file=no",
            "-n",
            (
                "--load=module-native-protocol-unix "
                f"socket={socket_path} auth-anonymous=1"
            ),
            (
                "--load=module-null-sink "
                f"sink_name={PULSE_SINK} rate=48000 channels=2"
            ),
        ]
        if os.geteuid() == 0:
            command.extend(["--system=true", "--disallow-module-loading=true"])
        with log_path.open("wb") as error_log:
            self._audio = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=error_log,
                env=environment,
                start_new_session=True,
            )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self._audio.poll() is not None:
                raise RuntimeError(
                    "PulseAudio exited during startup "
                    f"({self._audio.returncode}): {_read_log(log_path)}"
                )
            if socket_path.exists():
                return
            time.sleep(0.05)
        raise RuntimeError(
            "PulseAudio did not create its session socket: "
            f"{_read_log(log_path)}"
        )

    @staticmethod
    def _spawn(
        command: list[str], environment: dict[str, str]
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        )

    def _wait_for_display(self) -> XTestInput:
        deadline = time.monotonic() + 5
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._xvfb is not None and self._xvfb.poll() is not None:
                raise RuntimeError("Xvfb exited during startup")
            try:
                return XTestInput(self.display_name)
            except (OSError, RuntimeError) as error:
                last_error = error
                time.sleep(0.05)
        raise RuntimeError("Xvfb did not become ready") from last_error

    @staticmethod
    def _wait_for_process(process: subprocess.Popen[bytes], name: str) -> None:
        time.sleep(0.25)
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"{name} exited during startup ({return_code})")

    def apply_input(self, message: dict[str, Any]) -> None:
        try:
            sequence = int(message.get("seq", -1))
        except (TypeError, ValueError):
            return
        if sequence < self._last_sequence or self._input is None:
            return

        keys = message.get("keys", [])
        if not isinstance(keys, list):
            keys = []
        desired = {
            KEY_MAP[key] for key in keys if isinstance(key, str) and key in KEY_MAP
        }
        gamepad = message.get("gamepad")
        if isinstance(gamepad, dict):
            desired.update(_gamepad_keysyms(gamepad))
        self._input.update(desired)
        mouse = message.get("mouse")
        if isinstance(mouse, dict):
            self._input.mouse(
                _bounded_int(mouse.get("dx"), -2048, 2048),
                _bounded_int(mouse.get("dy"), -2048, 2048),
                _mouse_buttons(mouse.get("buttons")),
            )
        self._last_sequence = sequence

    def restart(self) -> None:
        if self._input is None:
            return
        self._input.update({"Escape"})
        self._input.update(set())

    def read_rgb(self) -> bytes:
        if self._capture is None or self._capture.stdout is None:
            raise RuntimeError("Doom capture is not running")
        frame = bytearray(FRAME_BYTES)
        view = memoryview(frame)
        offset = 0
        while offset < FRAME_BYTES:
            chunk = self._capture.stdout.read(FRAME_BYTES - offset)
            if not chunk:
                return_code = self._capture.poll()
                raise RuntimeError(f"Doom capture ended ({return_code})")
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        return bytes(frame)

    def snapshot(self) -> dict[str, int | str]:
        game_state = "running"
        if self._game is not None and self._game.poll() is not None:
            game_state = "exited"
        return {
            "backend": self.mode,
            "phase": game_state,
            "input_seq": self._last_sequence,
            "uptime_seconds": round(time.monotonic() - self._started_at),
        }

    def close(self) -> None:
        if self._input is not None:
            try:
                self._input.close()
            except (OSError, RuntimeError):
                pass
            self._input = None
        for process in (self._capture, self._game, self._xvfb, self._audio):
            try:
                _stop_process(process)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._capture = None
        self._game = None
        self._xvfb = None
        self._audio = None
        self.pulse_server = None
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None


def _gamepad_keysyms(gamepad: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    axes = gamepad.get("axes")
    if isinstance(axes, list):
        horizontal = _axis(axes, 0)
        vertical = _axis(axes, 1)
        if horizontal < -0.25:
            result.add("Left")
        elif horizontal > 0.25:
            result.add("Right")
        if vertical < -0.25:
            result.add("Up")
        elif vertical > 0.25:
            result.add("Down")
    buttons = gamepad.get("buttons")
    if isinstance(buttons, list):
        if _button(buttons, 0):
            result.add("Control_L")
        if _button(buttons, 1):
            result.add("space")
        if _button(buttons, 4) or _button(buttons, 5):
            result.add("Shift_L")
    return result


def _axis(values: list[Any], index: int) -> float:
    if index >= len(values):
        return 0.0
    try:
        return max(-1.0, min(1.0, float(values[index])))
    except (TypeError, ValueError):
        return 0.0


def _button(values: list[Any], index: int) -> bool:
    return index < len(values) and bool(values[index])


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return 0


def _mouse_buttons(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    buttons = set()
    for item in value:
        try:
            button = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= button <= 5:
            buttons.add(button)
    return buttons


def _prepare_audio_runtime(workspace: Path) -> Path:
    runtime_dir = workspace / "runtime"
    runtime_dir.mkdir()
    runtime_dir.chmod(0o777 if os.geteuid() == 0 else 0o700)
    return runtime_dir


def _read_log(path: Path) -> str:
    try:
        contents = path.read_text(errors="replace").strip()
    except OSError as error:
        return f"diagnostic log unavailable ({error})"
    if not contents:
        return "no diagnostic output"
    return contents[-2000:]


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _find_crispy_doom() -> str | None:
    executable = shutil.which("crispy-doom")
    if executable is not None:
        return executable
    return next((str(path) for path in CRISPY_DOOM_PATHS if path.is_file()), None)


def create_game_backend(
    mode: str | None = None, *, capture: bool = True, audio: bool = False
) -> OrbitBackend | DoomSession:
    default = "doom" if os.getenv("IS_ISOLATE_AGENT") else "auto"
    selected = (mode or os.getenv("FAL_CLOUD_GAME_BACKEND", default)).lower()
    if selected not in {"auto", "doom", "orbit"}:
        raise ValueError("FAL_CLOUD_GAME_BACKEND must be auto, doom, or orbit")
    if selected == "orbit":
        return OrbitBackend()
    if selected == "doom":
        DoomSession.require_available()
        backend = DoomSession()
        backend.start(capture=capture, audio=audio)
        return backend
    if DoomSession.is_available():
        backend = DoomSession()
        backend.start(capture=capture, audio=audio)
        return backend
    return OrbitBackend()
