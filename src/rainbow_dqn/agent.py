import torch
import numpy as np


from .network import DuelingDQN
from .replay_buffer import PrioritizedReplayBuffer, Experience, NStepBuffer


class RainbowAgent:
    def __init__(
        self,
        buffer_capacity: int,
        buffer_alpha: float,
        state_dim: int,
        num_actions: int,
        dense_size: int,
        learning_rate: float,
        gamma: float,
        n_steps: int,
        batch_size: int,
        update_target_every: int,
    ) -> None:
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, buffer_alpha)
        self.n_step_buffer = NStepBuffer(n_steps, gamma)
        self.online_network = DuelingDQN(state_dim, num_actions, dense_size)
        self.target_network = DuelingDQN(state_dim, num_actions, dense_size)

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate
        )
        self.gamma = gamma
        self.n_steps = n_steps

        self.gamma_n = gamma**n_steps
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.num_actions = num_actions
        self.step_count = 0

    def select_action(self, state: torch.Tensor) -> int:
        q_values = self.online_network(state)
        return int(torch.argmax(q_values))

    def store_experience(self, experience: Experience) -> None:
        self.n_step_buffer.add(experience)

        if self.n_step_buffer.is_ready():
            n_step_experience = self.n_step_buffer.get()
            self.replay_buffer.add(n_step_experience)

    def _update_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def _compute_loss(self, online: DuelingDQN, batch: dict) -> torch.Tensor:
        q_online = online(batch["states"])
        best_actions = torch.argmax(q_online, dim=-1)

        with torch.no_grad():
            q_target = self.target_network(batch["next_states"])
            q_target = q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = batch["rewards"] + self.gamma_n * q_target * (1.0 - batch["dones"])

        q_online_taken = q_online.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        td_errors = torch.abs(q_online_taken - target)
        loss = torch.mean(batch["weights"] * td_errors**2)

        return loss

    def _update_online_network(self, batch: dict) -> tuple:
        self.optimizer.zero_grad()
        loss = self._compute_loss(self.online_network, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        self.optimizer.step()

        with torch.no_grad():
            q_online = self.online_network(batch["states"])
            q_target = self.target_network(batch["next_states"])
            target = batch["rewards"] + self.gamma_n * q_target.max(dim=-1).values * (
                1.0 - batch["dones"]
            )
            q_online_taken = q_online.gather(1, batch["actions"].unsqueeze(1)).squeeze(
                1
            )
            td_errors = torch.abs(q_online_taken - target)

        return loss.item(), td_errors

    def train_step(self, beta: float) -> float | None:
        if not self.replay_buffer.is_ready(self.batch_size):
            return

        batch = self.replay_buffer.sample(self.batch_size, beta)
        loss, td_errors = self._update_online_network(batch)

        self.replay_buffer.update_priorities(batch["indexes"], np.array(td_errors))

        if self.step_count % self.update_target_every == 0:
            self._update_target_network()
        self.step_count += 1

        return loss
