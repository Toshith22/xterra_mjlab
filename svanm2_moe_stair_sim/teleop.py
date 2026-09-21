"""Keyboard teleoperation and track teleport manager for SvanM2 stair simulation.

Enforces hardware speed ceilings:
- vx (forward/backward): ±0.5 m/s
- vy (lateral): ±0.6 m/s
- yaw_rate (turning): ±0.4 rad/s

Dual-input support:
1. Terminal non-blocking keyboard listener (no GLFW conflicts whatsoever).
2. MuJoCo viewer GLFW key callback with non-conflicting arrow keys and WASD fallback.
"""

import os
import sys
import threading
import numpy as np

# Hardware-verified speed limits
MAX_VX = 0.50
MAX_VY = 0.60
MAX_YAW = 0.40

STEP_VX = 0.05
STEP_VY = 0.05
STEP_YAW = 0.05

SPAWN_HEIGHT = 0.365  # Base height ensuring feet rest cleanly on ground without sinking


class KeyboardTeleop:
    def __init__(self, controller, track_configs, track_spacing: float = 3.80):
        self.controller = controller
        self.track_configs = track_configs
        self.track_spacing = track_spacing
        
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0
        self.current_track = 1  # 1 to 6
        self.status_msg = "Ready. Press W or Up Arrow to walk."
        self._listener = None

    @property
    def command(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.yaw_rate)

    def stop(self):
        """Zero all command velocities."""
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0
        self.status_msg = "Commands zeroed (standing hold)."

    def get_track_y(self, track_idx_1based: int) -> float:
        return (track_idx_1based - 1 - 2.5) * self.track_spacing

    def teleport_to_track(self, track_idx_1based: int, on_top: bool = False):
        """Teleport robot to specified track."""
        if not (1 <= track_idx_1based <= len(self.track_configs)):
            return
        self.current_track = track_idx_1based
        cfg = self.track_configs[track_idx_1based - 1]
        y = self.get_track_y(track_idx_1based)

        if on_top:
            # Top platform: X = 3.6 + 1.5 = 5.1m, Z = 12 * h + SPAWN_HEIGHT, facing -X (yaw = pi)
            x = 5.10
            z = 12 * cfg["height"] + SPAWN_HEIGHT
            yaw = np.pi  # Facing towards the stairs (downstairs direction)
            self.status_msg = f"Teleported to Top Platform of Track {track_idx_1based} ({cfg['label']}), facing downstairs."
        else:
            # Approach runway: X = -1.5m, Z = SPAWN_HEIGHT, facing +X (yaw = 0)
            x = -1.50
            z = SPAWN_HEIGHT
            yaw = 0.0
            self.status_msg = f"Teleported to Entrance of Track {track_idx_1based} ({cfg['label']}), facing stairs."

        self.stop()
        self.controller.reset(target_pos=(x, y, z), yaw=yaw)

    def handle_char(self, ch: str):
        """Handle character input from terminal stdin."""
        c = ch.lower()
        if c == 'w':
            self.vx = min(MAX_VX, self.vx + STEP_VX)
            self.status_msg = f"Forward: vx = {self.vx:+.2f} m/s"
        elif c == 's':
            self.vx = max(-MAX_VX, self.vx - STEP_VX)
            self.status_msg = f"Backward: vx = {self.vx:+.2f} m/s"
        elif c == 'a':
            self.vy = min(MAX_VY, self.vy + STEP_VY)
            self.status_msg = f"Strafe Left: vy = {self.vy:+.2f} m/s"
        elif c == 'd':
            self.vy = max(-MAX_VY, self.vy - STEP_VY)
            self.status_msg = f"Strafe Right: vy = {self.vy:+.2f} m/s"
        elif c == 'q':
            self.yaw_rate = min(MAX_YAW, self.yaw_rate + STEP_YAW)
            self.status_msg = f"Turn Left: yaw = {self.yaw_rate:+.2f} rad/s"
        elif c == 'e':
            self.yaw_rate = max(-MAX_YAW, self.yaw_rate - STEP_YAW)
            self.status_msg = f"Turn Right: yaw = {self.yaw_rate:+.2f} rad/s"
        elif c == ' ':
            self.stop()
        elif c in ('1', '2', '3', '4', '5', '6'):
            self.teleport_to_track(int(c), on_top=False)
        elif c == 't':
            self.teleport_to_track(self.current_track, on_top=True)
        elif c == 'r':
            self.teleport_to_track(self.current_track, on_top=False)

    def handle_key(self, keycode: int):
        """Handle GLFW / viewer key events."""
        # W / w / Up Arrow (265) / I / i
        if keycode in (ord('W'), ord('w'), 265, ord('I'), ord('i')):
            self.vx = min(MAX_VX, self.vx + STEP_VX)
            self.status_msg = f"Forward: vx = {self.vx:+.2f} m/s"
        # S / s / Down Arrow (264) / K / k
        elif keycode in (ord('S'), ord('s'), 264, ord('K'), ord('k')):
            self.vx = max(-MAX_VX, self.vx - STEP_VX)
            self.status_msg = f"Backward: vx = {self.vx:+.2f} m/s"
        # A / a / Left Arrow (263) / J / j
        elif keycode in (ord('A'), ord('a'), 263, ord('J'), ord('j')):
            self.vy = min(MAX_VY, self.vy + STEP_VY)
            self.status_msg = f"Strafe Left: vy = {self.vy:+.2f} m/s"
        # D / d / Right Arrow (262) / U / u
        elif keycode in (ord('D'), ord('d'), 262, ord('U'), ord('u')):
            self.vy = max(-MAX_VY, self.vy - STEP_VY)
            self.status_msg = f"Strafe Right: vy = {self.vy:+.2f} m/s"
        # Q / q: Turn Left
        elif keycode in (ord('Q'), ord('q')):
            self.yaw_rate = min(MAX_YAW, self.yaw_rate + STEP_YAW)
            self.status_msg = f"Turn Left: yaw = {self.yaw_rate:+.2f} rad/s"
        # E / e: Turn Right
        elif keycode in (ord('E'), ord('e')):
            self.yaw_rate = max(-MAX_YAW, self.yaw_rate - STEP_YAW)
            self.status_msg = f"Turn Right: yaw = {self.yaw_rate:+.2f} rad/s"
        # Space (32): STOP
        elif keycode == 32:
            self.stop()
        # Teleport keys 1-6 (ASCII 49-54)
        elif 49 <= keycode <= 54:
            track_num = keycode - 48
            self.teleport_to_track(track_num, on_top=False)
        # T / t: Teleport to Top Platform
        elif keycode in (ord('T'), ord('t')):
            self.teleport_to_track(self.current_track, on_top=True)
        # R / r: Reset on current track
        elif keycode in (ord('R'), ord('r')):
            self.teleport_to_track(self.current_track, on_top=False)

    def start_terminal_listener(self):
        """Start non-blocking terminal stdin listener in background thread."""
        if not sys.stdin.isatty():
            return
        self._listener = TerminalKeyboardListener(self)
        self._listener.start()

    def stop_terminal_listener(self):
        """Stop terminal listener cleanly."""
        if self._listener:
            self._listener.stop()
            self._listener = None

    def get_track_info(self, y_pos: float, x_pos: float) -> dict:
        """Infer current track and terrain region from robot position."""
        track_diffs = [abs(y_pos - self.get_track_y(i)) for i in range(1, 7)]
        closest_idx = int(np.argmin(track_diffs)) + 1
        cfg = self.track_configs[closest_idx - 1]

        if x_pos < 0.0:
            region = "Approach Runway"
        elif 0.0 <= x_pos <= 3.60:
            step_idx = int(x_pos / 0.30) + 1
            region = f"On Stairs (Step {min(step_idx, 12)}/12)"
        else:
            region = "Top Turnaround Platform"

        return {
            "track": closest_idx,
            "label": cfg["label"],
            "step_height_m": cfg["height"],
            "region": region,
        }


class TerminalKeyboardListener:
    """Reads characters non-blockingly from stdin without needing Enter."""
    def __init__(self, teleop: KeyboardTeleop):
        self.teleop = teleop
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.2)

    def _run(self):
        import tty
        import termios
        import select

        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except Exception:
            return

        try:
            tty.setcbreak(fd)
            while self.running:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.teleop.handle_char(ch)
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
