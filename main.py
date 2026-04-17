from src.viz.renderer import TrafficRenderer
from src.environment.traffic_env import TrafficEnv
import time

env = TrafficEnv()
renderer = TrafficRenderer(render_every=10)  # renderuj co 10 kroków

obs, _ = env.reset()
for step in range(100000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    renderer.render(obs, reward, episode=0, total_steps=step)
    time.sleep(0.1)


renderer.close()
