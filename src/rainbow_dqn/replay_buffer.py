import random
import numpy as np
from collections import namedtuple

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "next_state", "done"],
)


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.priority_sum = [0.0 for _ in range(2 * capacity + 1)]
        self.priority_min = [float("inf") for _ in range(2 * capacity + 1)]

    def set_priority(self, idx: int, priority_alpha: float) -> None:
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx: int, priority_alpha: float) -> None:
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx > 1:
            idx //= 2
            self.priority_min[idx] = min(
                self.priority_min[2 * idx], self.priority_min[2 * idx + 1]
            )

    def _set_priority_sum(self, idx: int, priority_alpha: float) -> None:
        idx += self.capacity
        self.priority_sum[idx] = priority_alpha
        while idx > 1:
            idx //= 2
            self.priority_sum[idx] = (
                self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
            )

    def find_prefix_sum_idx(self, prefix_sum: float) -> int:
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[2 * idx] > prefix_sum:
                idx *= 2
            else:
                prefix_sum -= self.priority_sum[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity

    @property
    def sum(self) -> float:
        return self.priority_sum[1]

    @property
    def min(self) -> float:
        return self.priority_min[1]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.data = [None for _ in range(capacity)]
        self.next_idx = 0
        self.size = 0

    def add(self, experience: Experience) -> None:
        idx = self.next_idx
        self.data[idx] = experience
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        priority_alpha = self.max_priority**self.alpha
        self.tree.set_priority(idx, priority_alpha)

    def sample(self, batch_size: int, beta: float) -> dict:
        indexes = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            p = random.random() * self.tree.sum
            indexes[i] = self.tree.find_prefix_sum_idx(p)

        prob_min = self.tree.min / self.tree.sum
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = indexes[i]
            prob = self.tree.priority_sum[idx + self.capacity] / self.tree.sum
            weight = (prob * self.size) ** (-beta)
            weights[i] = weight / max_weight

        experiences = [self.data[idx] for idx in indexes]

        return {
            "states": np.array([e.state for e in experiences], dtype=np.float32),
            "actions": np.array([e.action for e in experiences], dtype=np.int32),
            "rewards": np.array([e.reward for e in experiences], dtype=np.float32),
            "next_states": np.array(
                [e.next_state for e in experiences], dtype=np.float32
            ),
            "dones": np.array([e.done for e in experiences], dtype=np.float32),
            "weights": weights,
            "indexes": indexes,
        }

    def update_priorities(self, indexes: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indexes, priorities):
            self.tree.set_priority(idx, priority**self.alpha)
            self.max_priority = max(self.max_priority, priority)

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size
