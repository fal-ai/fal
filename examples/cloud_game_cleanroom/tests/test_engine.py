from __future__ import annotations

import unittest

from engine import HEIGHT, WIDTH, InputState, OrbitBreaker


class InputStateTests(unittest.TestCase):
    def test_keyboard_and_gamepad_are_combined(self) -> None:
        state = InputState.from_message(
            {
                "seq": 14,
                "keys": ["KeyA", "Space"],
                "gamepad": {
                    "axes": [0.8],
                    "buttons": [False, False, False, False, False, False],
                },
            }
        )

        self.assertTrue(state.left)
        self.assertTrue(state.right)
        self.assertTrue(state.launch)
        self.assertEqual(state.sequence, 14)

    def test_invalid_payload_falls_back_to_neutral_input(self) -> None:
        state = InputState.from_message(
            {"seq": "bad", "keys": "KeyA", "gamepad": {"axes": ["bad"]}}
        )

        self.assertFalse(state.left)
        self.assertFalse(state.right)
        self.assertFalse(state.launch)
        self.assertEqual(state.sequence, -1)


class OrbitBreakerTests(unittest.TestCase):
    def test_ball_launches_on_rising_edge(self) -> None:
        game = OrbitBreaker(seed=2)
        game.apply_input({"seq": 1, "keys": ["Space"]})

        game.advance(1 / 30)

        self.assertEqual(game.phase, "playing")
        self.assertLess(game.ball_vy, 0)

    def test_paddle_stays_inside_playfield(self) -> None:
        game = OrbitBreaker()
        game.apply_input({"seq": 1, "keys": ["ArrowLeft"]})

        for _ in range(100):
            game.advance(0.05)

        self.assertGreaterEqual(game.paddle_x, 20)

    def test_missing_ball_consumes_life(self) -> None:
        game = OrbitBreaker()
        game.phase = "playing"
        game.ball_y = HEIGHT + game.ball_radius + 1
        game.ball_vy = 20

        game.advance(0.01)

        self.assertEqual(game.lives, 2)
        self.assertEqual(game.phase, "ready")

    def test_brick_collision_scores_and_removes_brick(self) -> None:
        game = OrbitBreaker()
        brick = game.bricks[0]
        game.phase = "playing"
        game.ball_x = brick.x + brick.width / 2
        game.ball_y = brick.y - game.ball_radius - 0.1
        game.ball_vx = 0
        game.ball_vy = 80

        game.advance(0.01)

        self.assertFalse(brick.alive)
        self.assertEqual(game.score, 120)
        self.assertLess(game.ball_vy, 0)

    def test_render_has_expected_rgb_size(self) -> None:
        frame = OrbitBreaker().render_rgb()

        self.assertEqual(len(frame), WIDTH * HEIGHT * 3)
        self.assertNotEqual(len(set(frame)), 1)

    def test_older_input_sequence_is_ignored(self) -> None:
        game = OrbitBreaker()
        game.apply_input({"seq": 10, "keys": ["ArrowRight"]})
        game.apply_input({"seq": 9, "keys": ["ArrowLeft"]})

        self.assertTrue(game.input.right)
        self.assertFalse(game.input.left)


if __name__ == "__main__":
    unittest.main()
