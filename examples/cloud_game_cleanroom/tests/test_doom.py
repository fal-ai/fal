from __future__ import annotations

import tempfile
import unittest
from os import environ
from pathlib import Path
from stat import S_IMODE
from typing import Any
from unittest.mock import Mock, patch

from doom import (
    FRAME_BYTES,
    DoomSession,
    OrbitBackend,
    _gamepad_keysyms,
    _prepare_audio_runtime,
    create_game_backend,
)


class DoomInputTests(unittest.TestCase):
    def test_keyboard_state_is_mapped_and_old_sequences_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wad = Path(directory) / "freedoom1.wad"
            wad.touch()
            session = DoomSession(executable="/usr/bin/crispy-doom", wad_path=wad)
            input_device = Mock()
            session._input = input_device

            session.apply_input(
                {
                    "seq": 8,
                    "keys": ["KeyW", "KeyA", "ControlLeft", "unknown"],
                }
            )
            session.apply_input({"seq": 7, "keys": ["KeyS"]})

        input_device.update.assert_called_once_with({"Up", "comma", "Control_L"})
        self.assertEqual(session.snapshot()["input_seq"], 8)

    def test_gamepad_maps_movement_actions_and_run(self) -> None:
        keysyms = _gamepad_keysyms(
            {
                "axes": [0.8, -0.9],
                "buttons": [True, True, False, False, True],
            }
        )

        self.assertEqual(keysyms, {"Right", "Up", "Control_L", "space", "Shift_L"})

    def test_relative_mouse_input_is_bounded_and_buttons_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wad = Path(directory) / "freedoom1.wad"
            wad.touch()
            session = DoomSession(executable="/usr/bin/crispy-doom", wad_path=wad)
            input_device = Mock()
            session._input = input_device

            session.apply_input(
                {
                    "seq": 1,
                    "keys": [],
                    "mouse": {
                        "dx": 999999,
                        "dy": -999999,
                        "buttons": [1, 3, 99, "invalid"],
                    },
                }
            )

        input_device.mouse.assert_called_once_with(2048, -2048, {1, 3})


class DoomCaptureTests(unittest.TestCase):
    def test_capture_reads_a_complete_frame_across_short_chunks(self) -> None:
        session = DoomSession(executable="/usr/bin/crispy-doom")
        process = Mock()
        process.stdout.read.side_effect = [
            b"\x01" * 17,
            b"\x02" * (FRAME_BYTES - 17),
        ]
        session._capture = process

        frame = session.read_rgb()

        self.assertEqual(len(frame), FRAME_BYTES)
        self.assertEqual(frame[:17], b"\x01" * 17)
        self.assertEqual(frame[17:], b"\x02" * (FRAME_BYTES - 17))


class DoomAudioTests(unittest.TestCase):
    @patch("doom.os.geteuid", return_value=0)
    def test_root_runtime_is_writable_after_umask_is_applied(self, _: Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = _prepare_audio_runtime(Path(directory))

            self.assertEqual(S_IMODE(runtime_dir.stat().st_mode), 0o777)

    @patch("doom.time.sleep")
    @patch("doom.time.monotonic", return_value=0)
    @patch("doom.subprocess.Popen")
    @patch("doom.os.geteuid", return_value=0)
    def test_root_audio_starts_with_session_socket_and_system_mode(
        self,
        _geteuid: Mock,
        popen: Mock,
        _monotonic: Mock,
        sleep: Mock,
    ) -> None:
        process = popen.return_value
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = Path(directory)
            sleep.side_effect = lambda _: runtime_dir.joinpath("pulse.sock").touch()
            session = DoomSession(executable="/usr/bin/crispy-doom")

            session._start_audio({"XDG_RUNTIME_DIR": directory})

        command = popen.call_args.args[0]
        self.assertIn("--system=true", command)
        self.assertIn("--disallow-module-loading=true", command)
        self.assertIn(
            f"--load=module-native-protocol-unix "
            f"socket={runtime_dir}/pulse.sock auth-anonymous=1",
            command,
        )
        self.assertEqual(session.pulse_server, f"unix:{runtime_dir}/pulse.sock")

    @patch("doom.time.sleep")
    @patch("doom.time.monotonic", return_value=0)
    @patch("doom.subprocess.Popen")
    @patch("doom.os.geteuid", return_value=1000)
    def test_non_root_audio_omits_system_mode(
        self,
        _geteuid: Mock,
        popen: Mock,
        _monotonic: Mock,
        sleep: Mock,
    ) -> None:
        process = popen.return_value
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = Path(directory)
            sleep.side_effect = lambda _: runtime_dir.joinpath("pulse.sock").touch()
            session = DoomSession(executable="/usr/bin/crispy-doom")

            session._start_audio({"XDG_RUNTIME_DIR": directory})

        command = popen.call_args.args[0]
        self.assertNotIn("--system=true", command)
        self.assertNotIn("--disallow-module-loading=true", command)

    @patch("doom.subprocess.Popen")
    def test_audio_early_exit_includes_diagnostic_log(self, popen: Mock) -> None:
        process = popen.return_value
        process.poll.return_value = 1
        process.returncode = 1

        def write_diagnostic(*_args: Any, **kwargs: Any) -> Mock:
            error_log = kwargs["stderr"]
            error_log.write(b"module-native-protocol-unix failed\n")
            error_log.flush()
            return process

        popen.side_effect = write_diagnostic
        with tempfile.TemporaryDirectory() as directory:
            session = DoomSession(executable="/usr/bin/crispy-doom")

            with self.assertRaisesRegex(
                RuntimeError, "module-native-protocol-unix failed"
            ):
                session._start_audio({"XDG_RUNTIME_DIR": directory})

    @patch("doom.time.monotonic", side_effect=[0, 6])
    @patch("doom.subprocess.Popen")
    @patch("doom.os.geteuid", return_value=0)
    def test_audio_timeout_is_cleaned_up_by_session_start(
        self, _geteuid: Mock, popen: Mock, _monotonic: Mock
    ) -> None:
        process = popen.return_value
        process.poll.return_value = None
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            wad = Path(directory) / "freedoom1.wad"
            wad.touch()
            session = DoomSession(
                executable="/usr/bin/crispy-doom",
                wad_path=wad,
            )

            with self.assertRaisesRegex(
                RuntimeError, "did not create its session socket"
            ):
                session.start(capture=False, audio=True)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=2)
        self.assertIsNone(session._audio)
        self.assertIsNone(session._workspace)


class BackendSelectionTests(unittest.TestCase):
    def test_orbit_mode_is_an_explicit_local_fallback(self) -> None:
        backend = create_game_backend("orbit")

        self.assertIsInstance(backend, OrbitBackend)
        self.assertEqual(len(backend.read_rgb()), FRAME_BYTES)

    @patch("doom.DoomSession.is_available", return_value=False)
    def test_auto_mode_falls_back_when_runtime_is_missing(self, _: Mock) -> None:
        self.assertIsInstance(create_game_backend("auto"), OrbitBackend)

    @patch("doom.DoomSession.availability_issues")
    def test_runner_default_requires_doom(self, availability_issues: Mock) -> None:
        availability_issues.return_value = ["missing executable: crispy-doom"]
        with patch.dict(environ, {"IS_ISOLATE_AGENT": "1"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "crispy-doom"):
                create_game_backend()

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto, doom, or orbit"):
            create_game_backend("quake")


class ContainerTests(unittest.TestCase):
    def test_container_pins_freely_licensed_game_versions(self) -> None:
        dockerfile = Path(__file__).parents[1].joinpath("Dockerfile").read_text()

        self.assertIn("CRISPY_DOOM_VERSION=7.1", dockerfile)
        self.assertIn(
            "f0eb02afb81780165ddc81583ed5648cbee8b3205bcc27e181b3f61eb26f8416",
            dockerfile,
        )
        self.assertIn("freedoom=0.13.0-2", dockerfile)
        self.assertIn("gstreamer1.0-plugins-bad", dockerfile)
        self.assertIn("gstreamer1.0-nice", dockerfile)
        self.assertIn("libnvrtc12=12.4.127~12.4.1-2", dockerfile)
        self.assertIn(
            "ln -s libnvrtc.so.12 /usr/lib/x86_64-linux-gnu/libnvrtc.so",
            dockerfile,
        )
        self.assertIn("python3-gst-1.0", dockerfile)
        self.assertIn("pulseaudio", dockerfile)
        self.assertIn("/tmp/crispy-doom/build/src/crispy-doom", dockerfile)
        self.assertIn("test -x /opt/crispy-doom/bin/crispy-doom", dockerfile)
        self.assertIn("crispy-doom-7.1.tar.gz", dockerfile)
        self.assertNotIn("doom1.wad", dockerfile.lower())


if __name__ == "__main__":
    unittest.main()
