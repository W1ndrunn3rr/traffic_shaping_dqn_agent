import numpy as np
import wandb
import hydra
from omegaconf import DictConfig
from flax import nnx
import jax.numpy as jnp
import os
import orbax.checkpoint as ocp


from src.rainbow_dqn.agent import RainbowAgent
from src.rainbow_dqn.replay_buffer import Experience
from src.environment.traffic_env import TrafficEnv
from src.viz.renderer import TrafficRenderer


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    checkpoint_dir = os.path.abspath(cfg.training.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    wandb.init(
        project=cfg.wandb.project,
        config={
            "agent": dict(cfg.agent),
            "env": dict(cfg.env),
            "training": dict(cfg.training),
        },
    )

    env = TrafficEnv(max_queue_len=cfg.env.max_queue_len)
    renderer = TrafficRenderer(render_every=cfg.renderer.render_every)

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

    beta_start = cfg.training.beta_start
    beta_frames = cfg.training.beta_frames

    for episode in range(cfg.training.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        truncated = False

        while not truncated:
            beta = beta_start + (1.0 - beta_start) * min(
                1.0, agent.step_count / beta_frames
            )

            action = agent.select_action(jnp.array(obs))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_experience(
                Experience(obs, action, reward, next_obs, terminated or truncated)
            )

            loss = agent.train_step(beta)

            if loss is not None:
                wandb.log({"loss": loss, "step": agent.step_count})

            obs = next_obs
            episode_reward += reward

            renderer.render(obs, episode_reward, episode, agent.step_count)

        wandb.log(
            {
                "episode_reward": episode_reward,
                "episode": episode,
                "step": agent.step_count,
                "beta": beta,
            }
        )

        if episode % cfg.training.save_every == 0:
            state = nnx.state(agent.online_network)
            checkpointer.save(f"{checkpoint_dir}/episode_{episode}", state)

    renderer.close()
    wandb.finish()


if __name__ == "__main__":
    train()
