from __future__ import annotations

import unittest

from doom import FRAME_BYTES, DoomSession


class DoomRuntimeSmokeTests(unittest.TestCase):
    def test_real_game_capture_input_and_teardown(self) -> None:
        DoomSession.require_available()
        session = DoomSession()
        processes = []
        try:
            session.start()
            processes = [session._capture, session._game, session._xvfb]
            first = session.read_rgb()
            session.apply_input({"seq": 1, "keys": ["ArrowUp", "ControlLeft"]})
            second = session.read_rgb()

            self.assertEqual(len(first), FRAME_BYTES)
            self.assertEqual(len(second), FRAME_BYTES)
            self.assertNotEqual(first, second)
            self.assertEqual(session.snapshot()["phase"], "running")
        finally:
            session.close()

        self.assertTrue(all(process is not None for process in processes))
        self.assertTrue(all(process.poll() is not None for process in processes))


if __name__ == "__main__":
    unittest.main()
