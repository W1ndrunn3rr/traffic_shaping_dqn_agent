import pygame
import numpy as np
from typing import Optional


WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (180, 180, 180)
DARK_GRAY = (60, 60, 60)
ROAD_COLOR = (50, 50, 50)
SIDEWALK_COLOR = (200, 195, 185)

RED_LIGHT = (220, 50, 50)
GREEN_LIGHT = (50, 200, 80)
YELLOW_LIGHT = (230, 190, 50)
LIGHT_OFF = (80, 80, 80)

CAR_NS = (70, 130, 200)
CAR_EW = (200, 100, 60)

PANEL_BG = (15, 15, 25)
ACCENT = (100, 180, 255)
TEXT_COLOR = (220, 220, 220)
SUBTEXT_COLOR = (140, 140, 160)


class TrafficRenderer:
    def __init__(self, width: int = 800, height: int = 600, render_every: int = 1):
        self.width = width
        self.height = height
        self.render_every = render_every
        self._step_count = 0

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font_large: Optional[pygame.font.Font] = None
        self.font_medium: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None
        self._initialized = False

        self.cx = width // 2
        self.cy = (height - 120) // 2 + 20
        self.road_w = 60
        self.intersection_size = 80

    def _init(self):
        pygame.init()
        pygame.display.set_caption("Traffic Light Controller — Rainbow DQN")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("menlo", 22, bold=True)
        self.font_medium = pygame.font.SysFont("menlo", 15)
        self.font_small = pygame.font.SysFont("menlo", 12)
        self._initialized = True

    def render(self, obs: np.ndarray, reward: float, episode: int, total_steps: int):
        self._step_count += 1
        if self._step_count % self.render_every != 0:
            return

        if not self._initialized:
            self._init()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        queue_N = int(obs[0])
        queue_S = int(obs[1])
        queue_E = int(obs[2])
        queue_W = int(obs[3])
        phase = int(obs[4])
        phase_dur = int(obs[5])
        sin_t = float(obs[6])
        cos_t = float(obs[7])

        hour = int((np.arctan2(sin_t, cos_t) / (2 * np.pi) * 24) % 24)
        minute = int((np.arctan2(sin_t, cos_t) / (2 * np.pi) * 24 % 1) * 60)

        self.screen.fill(BLACK)
        self._draw_info_panel(reward, episode, total_steps, phase_dur, hour, minute)
        self._draw_road()
        self._draw_intersection()
        self._draw_traffic_lights(phase)
        self._draw_queues(queue_N, queue_S, queue_E, queue_W, phase)
        self._draw_queue_bars(queue_N, queue_S, queue_E, queue_W)

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_road(self):
        cx, cy = self.cx, self.cy
        rw = self.road_w

        pygame.draw.rect(
            self.screen, ROAD_COLOR, (cx - rw // 2, 0, rw, self.height - 120)
        )
        pygame.draw.rect(self.screen, ROAD_COLOR, (0, cy - rw // 2, self.width, rw))

        pygame.draw.rect(
            self.screen, SIDEWALK_COLOR, (cx - rw // 2 - 6, 0, 6, self.height - 120)
        )
        pygame.draw.rect(
            self.screen, SIDEWALK_COLOR, (cx + rw // 2, 0, 6, self.height - 120)
        )
        pygame.draw.rect(
            self.screen, SIDEWALK_COLOR, (0, cy - rw // 2 - 6, self.width, 6)
        )
        pygame.draw.rect(self.screen, SIDEWALK_COLOR, (0, cy + rw // 2, self.width, 6))

        dash_len = 12
        gap_len = 10
        y = 0
        while y < cy - self.intersection_size // 2:
            pygame.draw.rect(self.screen, GRAY, (cx - 1, y, 2, dash_len))
            y += dash_len + gap_len

        y = cy + self.intersection_size // 2
        while y < self.height - 120:
            pygame.draw.rect(self.screen, GRAY, (cx - 1, y, 2, dash_len))
            y += dash_len + gap_len

        x = 0
        while x < cx - self.intersection_size // 2:
            pygame.draw.rect(self.screen, GRAY, (x, cy - 1, dash_len, 2))
            x += dash_len + gap_len

        x = cx + self.intersection_size // 2
        while x < self.width:
            pygame.draw.rect(self.screen, GRAY, (x, cy - 1, dash_len, 2))
            x += dash_len + gap_len

    def _draw_intersection(self):
        cx, cy = self.cx, self.cy
        s = self.intersection_size
        pygame.draw.rect(self.screen, ROAD_COLOR, (cx - s // 2, cy - s // 2, s, s))

    def _draw_traffic_lights(self, phase: int):
        cx, cy = self.cx, self.cy
        offset = self.intersection_size // 2 + 18

        ns_color = GREEN_LIGHT if phase == 0 else RED_LIGHT
        ew_color = GREEN_LIGHT if phase == 1 else RED_LIGHT

        positions = [
            (cx - offset, cy - offset, ns_color),
            (cx + offset - 10, cy + offset - 10, ns_color),
            (cx + offset - 10, cy - offset, ew_color),
            (cx - offset, cy + offset - 10, ew_color),
        ]

        for lx, ly, color in positions:
            pygame.draw.rect(
                self.screen, DARK_GRAY, (lx - 2, ly - 2, 14, 14), border_radius=3
            )
            pygame.draw.circle(self.screen, color, (lx + 5, ly + 5), 5)

    def _draw_queues(self, qN: int, qS: int, qE: int, qW: int, phase: int):
        cx, cy = self.cx, self.cy
        rw = self.road_w
        car_size = 10
        car_gap = 3
        max_visible = 6

        ns_color = CAR_NS
        ew_color = CAR_EW

        for i in range(min(qN, max_visible)):
            x = cx - car_size // 2
            y = cy - self.intersection_size // 2 - 20 - i * (car_size + car_gap)
            pygame.draw.rect(
                self.screen, ns_color, (x, y, car_size, car_size), border_radius=2
            )
        if qN > max_visible:
            self._draw_text(
                f"+{qN - max_visible}",
                cx,
                cy
                - self.intersection_size // 2
                - 20
                - max_visible * (car_size + car_gap)
                - 14,
                self.font_small,
                SUBTEXT_COLOR,
                center=True,
            )

        for i in range(min(qS, max_visible)):
            x = cx - car_size // 2
            y = cy + self.intersection_size // 2 + 20 + i * (car_size + car_gap)
            pygame.draw.rect(
                self.screen, ns_color, (x, y, car_size, car_size), border_radius=2
            )
        if qS > max_visible:
            self._draw_text(
                f"+{qS - max_visible}",
                cx,
                cy
                + self.intersection_size // 2
                + 20
                + max_visible * (car_size + car_gap)
                + 4,
                self.font_small,
                SUBTEXT_COLOR,
                center=True,
            )

        for i in range(min(qE, max_visible)):
            x = cx + self.intersection_size // 2 + 20 + i * (car_size + car_gap)
            y = cy - car_size // 2
            pygame.draw.rect(
                self.screen, ew_color, (x, y, car_size, car_size), border_radius=2
            )
        if qE > max_visible:
            self._draw_text(
                f"+{qE - max_visible}",
                cx
                + self.intersection_size // 2
                + 20
                + max_visible * (car_size + car_gap)
                + 4,
                cy,
                self.font_small,
                SUBTEXT_COLOR,
                center=True,
            )

        for i in range(min(qW, max_visible)):
            x = cx - self.intersection_size // 2 - 20 - i * (car_size + car_gap)
            y = cy - car_size // 2
            pygame.draw.rect(
                self.screen, ew_color, (x, y, car_size, car_size), border_radius=2
            )
        if qW > max_visible:
            self._draw_text(
                f"+{qW - max_visible}",
                cx
                - self.intersection_size // 2
                - 20
                - max_visible * (car_size + car_gap)
                - 14,
                cy,
                self.font_small,
                SUBTEXT_COLOR,
                center=True,
            )

        label_offset = 30
        self._draw_text(
            "N",
            cx + self.road_w // 2 + 8,
            cy - self.intersection_size // 2 - label_offset,
            self.font_small,
            SUBTEXT_COLOR,
        )
        self._draw_text(
            "S",
            cx + self.road_w // 2 + 8,
            cy + self.intersection_size // 2 + label_offset,
            self.font_small,
            SUBTEXT_COLOR,
        )
        self._draw_text(
            "E",
            cx + self.intersection_size // 2 + label_offset,
            cy - self.road_w // 2 - 16,
            self.font_small,
            SUBTEXT_COLOR,
        )
        self._draw_text(
            "W",
            cx - self.intersection_size // 2 - label_offset - 8,
            cy - self.road_w // 2 - 16,
            self.font_small,
            SUBTEXT_COLOR,
        )

    def _draw_queue_bars(self, qN: int, qS: int, qE: int, qW: int):
        panel_y = self.height - 120
        pygame.draw.rect(self.screen, PANEL_BG, (0, panel_y, self.width, 120))
        pygame.draw.line(self.screen, DARK_GRAY, (0, panel_y), (self.width, panel_y), 1)

        bar_max = 20
        bar_h = 40
        bar_w = 80
        spacing = (self.width - 4 * bar_w) // 5
        y_bar = panel_y + 55

        dirs = [
            ("N", qN, CAR_NS),
            ("S", qS, CAR_NS),
            ("E", qE, CAR_EW),
            ("W", qW, CAR_EW),
        ]
        for i, (label, q, color) in enumerate(dirs):
            x = spacing + i * (bar_w + spacing)
            fill_w = int(bar_w * min(q, bar_max) / bar_max)

            pygame.draw.rect(
                self.screen,
                DARK_GRAY,
                (x, y_bar - bar_h, bar_w, bar_h),
                border_radius=4,
            )
            if fill_w > 0:
                fill_color = (
                    RED_LIGHT
                    if q >= bar_max * 0.8
                    else (YELLOW_LIGHT if q >= bar_max * 0.5 else color)
                )
                pygame.draw.rect(
                    self.screen,
                    fill_color,
                    (x, y_bar - bar_h, fill_w, bar_h),
                    border_radius=4,
                )

            self._draw_text(
                label,
                x + bar_w // 2,
                y_bar - bar_h - 16,
                self.font_small,
                SUBTEXT_COLOR,
                center=True,
            )
            self._draw_text(
                str(q),
                x + bar_w // 2,
                y_bar + 6,
                self.font_medium,
                TEXT_COLOR,
                center=True,
            )

    def _draw_info_panel(
        self,
        reward: float,
        episode: int,
        steps: int,
        phase_dur: int,
        hour: int,
        minute: int,
    ):
        self._draw_text(f"EP {episode:04d}", 14, 12, self.font_large, ACCENT)
        self._draw_text(f"STEP {steps:06d}", 14, 36, self.font_medium, SUBTEXT_COLOR)
        self._draw_text(f"REWARD  {reward:+.0f}", 14, 56, self.font_medium, TEXT_COLOR)
        self._draw_text(
            f"PHASE DUR  {phase_dur}", 14, 76, self.font_medium, SUBTEXT_COLOR
        )

        time_str = f"{hour:02d}:{minute:02d}"
        self._draw_text(
            time_str, self.width - 14, 12, self.font_large, ACCENT, right=True
        )

        rush = (7 <= hour <= 9) or (16 <= hour <= 19)
        period = (
            "RUSH HOUR" if rush else ("NIGHT" if hour < 6 or hour > 22 else "NORMAL")
        )
        period_color = (
            RED_LIGHT if rush else (SUBTEXT_COLOR if period == "NIGHT" else GREEN_LIGHT)
        )
        self._draw_text(
            period, self.width - 14, 36, self.font_small, period_color, right=True
        )

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        font,
        color,
        center: bool = False,
        right: bool = False,
    ):
        surface = font.render(text, True, color)
        if center:
            rect = surface.get_rect(center=(x, y))
        elif right:
            rect = surface.get_rect(topright=(x, y))
        else:
            rect = surface.get_rect(topleft=(x, y))
        self.screen.blit(surface, rect)

    def close(self):
        if self._initialized:
            pygame.quit()
            self._initialized = False
