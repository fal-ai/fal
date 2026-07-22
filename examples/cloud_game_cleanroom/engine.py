from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

WIDTH = 960
HEIGHT = 540

Color = tuple[int, int, int]

BACKGROUND: Color = (12, 14, 17)
SURFACE: Color = (28, 31, 36)
GRID: Color = (22, 25, 29)
FOREGROUND: Color = (226, 231, 228)
ACCENT: Color = (92, 194, 154)
DANGER: Color = (210, 103, 90)
BRICK_COLORS: tuple[Color, ...] = (
    (80, 133, 148),
    (85, 153, 139),
    (150, 151, 96),
    (169, 119, 94),
)


@dataclass
class InputState:
    left: bool = False
    right: bool = False
    launch: bool = False
    restart: bool = False
    sequence: int = -1

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> "InputState":
        keys = message.get("keys", [])
        if not isinstance(keys, list):
            keys = []
        key_set = {key for key in keys if isinstance(key, str)}

        gamepad = message.get("gamepad")
        axis = 0.0
        launch = "Space" in key_set
        restart = "KeyR" in key_set
        if isinstance(gamepad, dict):
            axes = gamepad.get("axes", [])
            if isinstance(axes, list) and axes:
                try:
                    axis = max(-1.0, min(1.0, float(axes[0])))
                except (TypeError, ValueError):
                    axis = 0.0
            buttons = gamepad.get("buttons", [])
            if isinstance(buttons, list):
                launch = launch or any(
                    bool(buttons[index]) for index in (0, 1) if index < len(buttons)
                )
                restart = restart or (len(buttons) > 9 and bool(buttons[9]))

        try:
            sequence = int(message.get("seq", -1))
        except (TypeError, ValueError):
            sequence = -1

        return cls(
            left="ArrowLeft" in key_set or "KeyA" in key_set or axis < -0.2,
            right="ArrowRight" in key_set or "KeyD" in key_set or axis > 0.2,
            launch=launch,
            restart=restart,
            sequence=sequence,
        )


@dataclass
class Brick:
    x: float
    y: float
    width: float
    height: float
    color: Color
    alive: bool = True


class OrbitBreaker:
    paddle_width = 132.0
    paddle_height = 14.0
    paddle_y = HEIGHT - 58.0
    paddle_speed = 660.0
    ball_radius = 8.0
    initial_ball_speed = 345.0

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self.input = InputState()
        self.score = 0
        self.lives = 3
        self.level = 1
        self.phase = "ready"
        self.paddle_x = (WIDTH - self.paddle_width) / 2
        self.ball_x = WIDTH / 2
        self.ball_y = self.paddle_y - self.ball_radius - 2
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.bricks = self._make_bricks()
        self._background = self._make_background()
        self._previous_launch = False
        self._previous_restart = False

    def _make_bricks(self) -> list[Brick]:
        columns = 11
        rows = 5
        gap = 8
        brick_width = 72
        brick_height = 20
        total_width = columns * brick_width + (columns - 1) * gap
        left = (WIDTH - total_width) / 2
        top = 84
        return [
            Brick(
                x=left + column * (brick_width + gap),
                y=top + row * (brick_height + gap),
                width=brick_width,
                height=brick_height,
                color=BRICK_COLORS[row % len(BRICK_COLORS)],
            )
            for row in range(rows)
            for column in range(columns)
        ]

    def _make_background(self) -> bytes:
        frame = bytearray(BACKGROUND * (WIDTH * HEIGHT))
        for x in range(0, WIDTH, 48):
            self._draw_rect(frame, x, 0, 1, HEIGHT, GRID)
        for y in range(0, HEIGHT, 48):
            self._draw_rect(frame, 0, y, WIDTH, 1, GRID)
        return bytes(frame)

    def apply_input(self, message: dict[str, Any]) -> None:
        incoming = InputState.from_message(message)
        if incoming.sequence >= self.input.sequence:
            self.input = incoming

    def reset(self) -> None:
        self.score = 0
        self.lives = 3
        self.level = 1
        self.bricks = self._make_bricks()
        self.paddle_x = (WIDTH - self.paddle_width) / 2
        self._park_ball()

    def _park_ball(self) -> None:
        self.phase = "ready"
        self.ball_x = self.paddle_x + self.paddle_width / 2
        self.ball_y = self.paddle_y - self.ball_radius - 2
        self.ball_vx = 0.0
        self.ball_vy = 0.0

    def _launch(self) -> None:
        angle = self._rng.uniform(-0.72, 0.72)
        speed = self.initial_ball_speed + (self.level - 1) * 24
        self.ball_vx = math.sin(angle) * speed
        self.ball_vy = -math.cos(angle) * speed
        self.phase = "playing"

    def advance(self, seconds: float) -> None:
        seconds = max(0.0, min(seconds, 0.05))
        direction = int(self.input.right) - int(self.input.left)
        self.paddle_x += direction * self.paddle_speed * seconds
        self.paddle_x = max(20.0, min(WIDTH - 20.0 - self.paddle_width, self.paddle_x))

        restart_pressed = self.input.restart and not self._previous_restart
        launch_pressed = self.input.launch and not self._previous_launch
        self._previous_restart = self.input.restart
        self._previous_launch = self.input.launch

        if restart_pressed:
            self.reset()
        if self.phase == "ready":
            self.ball_x = self.paddle_x + self.paddle_width / 2
            self.ball_y = self.paddle_y - self.ball_radius - 2
            if launch_pressed:
                self._launch()
            return
        if self.phase == "game_over":
            return

        steps = max(1, math.ceil(seconds / 0.008))
        step = seconds / steps
        for _ in range(steps):
            self._advance_ball(step)
            if self.phase != "playing":
                break

    def _advance_ball(self, seconds: float) -> None:
        previous_x = self.ball_x
        previous_y = self.ball_y
        self.ball_x += self.ball_vx * seconds
        self.ball_y += self.ball_vy * seconds

        if self.ball_x - self.ball_radius < 16:
            self.ball_x = 16 + self.ball_radius
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + self.ball_radius > WIDTH - 16:
            self.ball_x = WIDTH - 16 - self.ball_radius
            self.ball_vx = -abs(self.ball_vx)

        if self.ball_y - self.ball_radius < 16:
            self.ball_y = 16 + self.ball_radius
            self.ball_vy = abs(self.ball_vy)

        if (
            self.ball_vy > 0
            and previous_y + self.ball_radius <= self.paddle_y
            and self.ball_y + self.ball_radius >= self.paddle_y
            and self.paddle_x - self.ball_radius
            <= self.ball_x
            <= self.paddle_x + self.paddle_width + self.ball_radius
        ):
            offset = (self.ball_x - (self.paddle_x + self.paddle_width / 2)) / (
                self.paddle_width / 2
            )
            speed = min(640.0, math.hypot(self.ball_vx, self.ball_vy) * 1.035)
            self.ball_vx = speed * max(-0.88, min(0.88, offset))
            self.ball_vy = -math.sqrt(max(1.0, speed * speed - self.ball_vx**2))
            self.ball_y = self.paddle_y - self.ball_radius - 0.1

        for brick in self.bricks:
            if not brick.alive or not self._ball_intersects(brick):
                continue
            brick.alive = False
            self.score += 120
            horizontal_entry = (
                previous_x + self.ball_radius <= brick.x
                or previous_x - self.ball_radius >= brick.x + brick.width
            )
            if horizontal_entry:
                self.ball_vx *= -1
            else:
                self.ball_vy *= -1
            break

        if self.ball_y - self.ball_radius > HEIGHT:
            self.lives -= 1
            if self.lives <= 0:
                self.phase = "game_over"
            else:
                self._park_ball()

        if self.bricks and all(not brick.alive for brick in self.bricks):
            self.level += 1
            self.bricks = self._make_bricks()
            self._park_ball()

    def _ball_intersects(self, brick: Brick) -> bool:
        nearest_x = max(brick.x, min(self.ball_x, brick.x + brick.width))
        nearest_y = max(brick.y, min(self.ball_y, brick.y + brick.height))
        dx = self.ball_x - nearest_x
        dy = self.ball_y - nearest_y
        return dx * dx + dy * dy <= self.ball_radius * self.ball_radius

    def snapshot(self) -> dict[str, int | str]:
        return {
            "score": self.score,
            "lives": self.lives,
            "level": self.level,
            "phase": self.phase,
            "paddle_x": round(self.paddle_x),
        }

    def render_rgb(self) -> bytes:
        frame = bytearray(self._background)
        self._draw_rect(frame, 15, 15, WIDTH - 30, 2, SURFACE)
        self._draw_rect(frame, 15, 15, 2, HEIGHT - 30, SURFACE)
        self._draw_rect(frame, WIDTH - 17, 15, 2, HEIGHT - 30, SURFACE)

        for brick in self.bricks:
            if not brick.alive:
                continue
            self._draw_rect(
                frame,
                int(brick.x),
                int(brick.y),
                int(brick.width),
                int(brick.height),
                brick.color,
            )
            self._draw_rect(
                frame,
                int(brick.x),
                int(brick.y),
                int(brick.width),
                2,
                tuple(min(255, channel + 25) for channel in brick.color),
            )

        paddle_color = ACCENT if self.phase != "game_over" else DANGER
        self._draw_rect(
            frame,
            int(self.paddle_x),
            int(self.paddle_y),
            int(self.paddle_width),
            int(self.paddle_height),
            paddle_color,
        )
        self._draw_circle(
            frame,
            int(self.ball_x),
            int(self.ball_y),
            int(self.ball_radius),
            FOREGROUND,
        )
        return bytes(frame)

    @staticmethod
    def _draw_rect(
        frame: bytearray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Color,
    ) -> None:
        left = max(0, x)
        top = max(0, y)
        right = min(WIDTH, x + width)
        bottom = min(HEIGHT, y + height)
        if left >= right or top >= bottom:
            return
        row = bytes(color) * (right - left)
        for row_y in range(top, bottom):
            offset = (row_y * WIDTH + left) * 3
            frame[offset : offset + len(row)] = row

    @classmethod
    def _draw_circle(
        cls,
        frame: bytearray,
        center_x: int,
        center_y: int,
        radius: int,
        color: Color,
    ) -> None:
        for y in range(center_y - radius, center_y + radius + 1):
            dy = y - center_y
            span = int(math.sqrt(max(0, radius * radius - dy * dy)))
            cls._draw_rect(
                frame,
                center_x - span,
                y,
                span * 2 + 1,
                1,
                color,
            )
