# Traffic Light Controller — Rainbow DQN

Reinforcement learning agent controlling a single intersection traffic light, trained using the Rainbow DQN algorithm implemented in JAX.

---

## Project Structure

```
traffic-rainbow-dqn/
├── env/
│   ├── traffic_env.py        
│   └── traffic_patterns.py   
├── rainbow/
│   ├── agent.py              
│   ├── network.py            
│   ├── replay_buffer.py      
│   └── utils.py              
├── viz/
│   └── renderer.py           
├── train.py
├── evaluate.py
└── config.py
```

---

## Environment

Custom Gymnasium environment simulating a single four-way intersection. The agent controls a traffic light with two phases — North/South green or East/West green.

**Observation space** — Box(8,) containing:
- Queue lengths for N, S, E, W directions (0 to max_queue_len)
- Current phase (0 or 1)
- Current phase duration
- Sin and cos encoding of time of day

**Action space** — Discrete(2):
- 0 — North/South green, East/West red
- 1 — East/West green, North/South red

**Reward** — negative sum of all queue lengths at each step. The agent learns to minimize total waiting time across all directions.

**Traffic model** — arrival rates follow a double Gaussian pattern peaking at 08:00 (morning rush) and 17:00 (evening rush). Arrivals are sampled from a Poisson distribution. North/South traffic is heavier in the morning, East/West in the evening. One simulation step corresponds to one minute. A full episode covers 1440 steps (24 hours).

---

## Rainbow DQN Components

| Component | Role |
|---|---|
| Double DQN | Prevents Q-value overestimation under changing traffic patterns |
| Dueling Network | Separates state value V(s) from action advantage A(s,a) |
| Prioritized Replay | Samples rush hour experiences more frequently |
| Noisy Nets | Replaces epsilon-greedy with learned exploration |
| N-step Returns | Better credit assignment for delayed effects of signal changes |
| Distributional (C51) | Models full distribution of waiting times, not just the mean |

---

## Stack

| Library | Purpose |
|---|---|
| jax[cpu] | JIT compilation, autodiff, vmap |
| flax | Neural network (Dueling + Noisy) |
| optax | Adam optimizer |
| rlax | C51, n-step returns, PER helpers |
| gymnasium | Environment base class |
| pygame-ce | Visualization |
| wandb | Experiment tracking |
| numpy | Replay buffer |

---

## Installation

Requires Python 3.11+, [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/yourname/traffic-rainbow-dqn
cd traffic-rainbow-dqn
uv sync
```

For GPU (Linux only, CUDA 12):
```bash
uv add "jax[cuda12]"
```

---

## Usage

Run environment sanity check:
```bash
uv run python -m env.traffic_env
```

Visualize with random policy:
```bash
uv run python -m viz.renderer
```

Train:
```bash
uv run python train.py
```

Evaluate:
```bash
uv run python evaluate.py --checkpoint path/to/checkpoint
```

---

## References

Hessel et al., Rainbow: Combining Improvements in Deep Reinforcement Learning, AAAI 2018. https://arxiv.org/abs/1710.02298