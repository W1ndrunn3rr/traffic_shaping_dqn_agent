import numpy as np
import gymnasium as gym
from .traffic_patterns import get_arrival_rate


class TrafficEnv(gym.Env):
    def __init__(self, max_queue_len: int = 20, max_steps_size: int = 1440) -> None:
        super().__init__()

        self.max_queue_len = max_queue_len
        self.max_steps_size = max_steps_size

        self.queue_len = np.zeros(4, dtype=np.int32)
        self.current_phase = 0
        self.phase_duration = 0
        self.time_step = 0

        self.action_space = gym.spaces.Discrete(2)

        self.observation_space = gym.spaces.Box(
            low=np.array([0] * 4 + [0, 0, -1, -1], dtype=np.float32),
            high=np.array(
                [max_queue_len] * 4 + [1, max_queue_len, 1, 1], dtype=np.float32
            ),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        hour = (self.time_step % 1440) // 60
        sin_time = np.sin(2 * np.pi * hour / 24)
        cos_time = np.cos(2 * np.pi * hour / 24)
        return np.array(
            [
                *self.queue_len,
                self.current_phase,
                self.phase_duration,
                sin_time,
                cos_time,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        return {"queue_len": self.queue_len.copy(), "phase": self.current_phase}

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.queue_len = np.zeros(4, dtype=np.int32)
        self.current_phase = 0
        self.phase_duration = 0
        self.time_step = 0
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple:
        self.time_step += 1

        if action != self.current_phase:
            self.current_phase = action
            self.phase_duration = 0
        else:
            self.phase_duration = min(self.phase_duration + 1, self.max_queue_len)

        green_dirs = [0, 1] if action == 0 else [2, 3]
        red_dirs = [2, 3] if action == 0 else [0, 1]

        hour = (self.time_step % 1440) // 60
        arrival_rates = get_arrival_rate(hour)

        for d in green_dirs:
            arrival = self.np_random.poisson(arrival_rates[d])
            departure = min(self.np_random.integers(0, 5), self.queue_len[d])
            self.queue_len[d] = int(
                np.clip(self.queue_len[d] + arrival - departure, 0, self.max_queue_len)
            )

        for d in red_dirs:
            arrival = self.np_random.poisson(arrival_rates[d])
            self.queue_len[d] = int(
                np.clip(self.queue_len[d] + arrival, 0, self.max_queue_len)
            )

        reward = float(-np.sum(self.queue_len))

        terminated = False
        truncated = self.time_step >= self.max_steps_size

        return self._get_obs(), reward, terminated, truncated, self._get_info()
