import wandb


def download_model(artifact_name: str) -> str:
    api = wandb.Api()
    artifact = api.artifact(artifact_name, type="model")
    artifact_dir = artifact.download()
    return artifact_dir


if __name__ == "__main__":
    artifact_name = "best-model:latest"

    model_dir = download_model(artifact_name)
    print(f"Model downloaded to: {model_dir}")
