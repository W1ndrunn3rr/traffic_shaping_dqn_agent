import numpy as np
import jax.numpy as jnp
from flax import nnx
import hydra
import os
from omegaconf import DictConfig
import orbax.checkpoint as ocp
from collections import defaultdict

from src.environment.traffic_env import TrafficEnv
from src.rainbow_dqn.agent import RainbowAgent
from src.rainbow_dqn.network import DuelingDQN


def run_episode(env: TrafficEnv, policy, num_steps: int) -> dict:
    obs, _ = env.reset(seed=None)
    metrics = defaultdict(list)

    for step in range(num_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        metrics["rewards"].append(reward)
        metrics["queue_N"].append(info["queue_len"][0])
        metrics["queue_S"].append(info["queue_len"][1])
        metrics["queue_E"].append(info["queue_len"][2])
        metrics["queue_W"].append(info["queue_len"][3])

        hour = (env.time_step % 1440) // 60
        metrics["is_rush"].append(1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0)

        if terminated or truncated:
            break

    return metrics


def aggregate_metrics(metrics: dict) -> dict:
    rewards = np.array(metrics["rewards"])
    is_rush = np.array(metrics["is_rush"])
    is_night = 1 - is_rush

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_queue_N": float(np.mean(metrics["queue_N"])),
        "mean_queue_S": float(np.mean(metrics["queue_S"])),
        "mean_queue_E": float(np.mean(metrics["queue_E"])),
        "mean_queue_W": float(np.mean(metrics["queue_W"])),
        "rush_reward": float(np.mean(rewards[is_rush == 1]))
        if is_rush.sum() > 0
        else 0.0,
        "night_reward": float(np.mean(rewards[is_night == 1]))
        if is_night.sum() > 0
        else 0.0,
    }


def random_policy(obs) -> int:
    return np.random.choice([0, 1])


def load_agent(cfg: DictConfig, checkpoint_path: str) -> RainbowAgent:
    rngs = nnx.Rngs(params=0, noise=1)
    agent = RainbowAgent(
        buffer_capacity=cfg.agent.buffer_capacity,
        buffer_alpha=cfg.agent.buffer_alpha,
        state_dim=cfg.agent.state_dim,
        num_actions=cfg.agent.num_actions,
        dense_size=cfg.agent.dense_size,
        learning_rate=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        n_steps=cfg.agent.n_steps,
        batch_size=cfg.agent.batch_size,
        update_target_every=cfg.agent.update_target_every,
        rngs=rngs,
    )
    checkpointer = ocp.StandardCheckpointer()
    abstract_state = nnx.state(agent.online_network)
    state = checkpointer.restore(
        os.path.abspath(checkpoint_path),
        target=abstract_state,
    )
    nnx.update(agent.online_network, state)
    return agent


def agent_policy(agent: RainbowAgent):
    def policy(obs) -> int:
        return agent.select_action(jnp.array(obs))

    return policy


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def evaluate(cfg: DictConfig) -> None:
    env = TrafficEnv(
        max_queue_len=cfg.env.max_queue_len,
        max_steps_size=cfg.env.max_steps,
    )

    random_results = []
    for _ in range(cfg.eval.num_episodes):
        metrics = run_episode(env, random_policy, cfg.env.max_steps)
        random_results.append(aggregate_metrics(metrics))

    agent = load_agent(cfg, cfg.eval.checkpoint_path)
    agent_results = []
    for _ in range(cfg.eval.num_episodes):
        metrics = run_episode(env, agent_policy(agent), cfg.env.max_steps)
        agent_results.append(aggregate_metrics(metrics))

    keys = [
        "mean_reward",
        "mean_queue_N",
        "mean_queue_S",
        "mean_queue_E",
        "mean_queue_W",
        "rush_reward",
        "night_reward",
    ]
    for k in keys:
        rand = np.mean([r[k] for r in random_results])
        agent_val = np.mean([r[k] for r in agent_results])
        diff = agent_val - rand
        print(
            f"{k:20s}  random: {rand:8.2f}  agent: {agent_val:8.2f}  diff: {diff:+.2f}"
        )


if __name__ == "__main__":
    evaluate()
