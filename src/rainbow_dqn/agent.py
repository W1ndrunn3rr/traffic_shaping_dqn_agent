from flax import nnx
import jax.numpy as jnp
import optax
import numpy as np
import jax

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
        rngs: nnx.Rngs,
    ) -> None:
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, buffer_alpha)
        self.n_step_buffer = NStepBuffer(n_steps, gamma)
        self.online_network = DuelingDQN(state_dim, num_actions, dense_size, rngs)
        self.target_network = DuelingDQN(state_dim, num_actions, dense_size, rngs)
        self.rng_key = jax.random.PRNGKey(0)

        self.optimizer = nnx.Optimizer(
            self.online_network,
            optax.adam(learning_rate),
            wrt=nnx.Param,
        )

        self.gamma = gamma
        self.n_steps = n_steps

        self.gamma_n = gamma**n_steps
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.num_actions = num_actions
        self.step_count = 0

    def select_action(self, state: jnp.ndarray) -> int:
        self.rng_key, key = jax.random.split(self.rng_key)
        q_values = self.online_network(state, key)
        return int(jnp.argmax(q_values))

    def store_experience(self, experience: Experience) -> None:
        self.n_step_buffer.add(experience)

        if self.n_step_buffer.is_ready():
            n_step_experience = self.n_step_buffer.get()
            self.replay_buffer.add(n_step_experience)

        self.replay_buffer.add(experience)

    def _update_target_network(self) -> None:
        online_state = nnx.state(self.online_network)
        nnx.update(self.target_network, online_state)

    def _compute_loss(
        self, online: DuelingDQN, batch: dict
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self.rng_key, key1 = jax.random.split(self.rng_key)
        self.rng_key, key2 = jax.random.split(self.rng_key)

        q_online = online(batch["states"], key1)
        best_actions = jnp.argmax(q_online, axis=-1)

        q_target = self.target_network(batch["next_states"], key2)
        q_target = jnp.take_along_axis(
            q_target, best_actions[:, None], axis=-1
        ).squeeze(-1)

        target = batch["rewards"] + self.gamma_n * q_target * (1.0 - batch["dones"])

        q_online_taken = jnp.take_along_axis(
            q_online, batch["actions"][:, None], axis=-1
        ).squeeze(-1)
        td_errors = jnp.abs(q_online_taken - target)
        loss = jnp.mean(batch["weights"] * td_errors**2)
        return loss

    def _update_online_network(self, batch: dict) -> tuple:
        loss, grads = nnx.value_and_grad(self._compute_loss, argnums=0)(
            self.online_network, batch
        )
        self.optimizer.update(self.online_network, grads)
        optax.clip_by_global_norm(10.0)

        self.rng_key, key1 = jax.random.split(self.rng_key)
        self.rng_key, key2 = jax.random.split(self.rng_key)
        q_online = self.online_network(batch["states"], key1)
        best_actions = jnp.argmax(q_online, axis=-1)
        q_target = self.target_network(batch["next_states"], key2)
        q_target = jnp.take_along_axis(
            q_target, best_actions[:, None], axis=-1
        ).squeeze(-1)
        target = jax.lax.stop_gradient(
            batch["rewards"] + self.gamma_n * q_target * (1.0 - batch["dones"])
        )
        q_online_taken = jnp.take_along_axis(
            q_online, batch["actions"][:, None], axis=-1
        ).squeeze(-1)
        td_errors = jnp.abs(q_online_taken - target)

        return float(loss), td_errors

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
