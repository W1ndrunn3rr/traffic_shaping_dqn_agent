train:
    PYTHONPATH=. uv run python scripts/train.py

eval checkpoint:
    PYTHONPATH=. uv run python scripts/evaluate.py eval.checkpoint_path={{checkpoint}}

render checkpoint:
    PYTHONPATH=. uv run python scripts/evaluate.py eval.checkpoint_path={{checkpoint}} renderer.render_every=1

clean:
    rm -rf checkpoints/* wandb/* outputs/* multirun/*

install:
    uv sync