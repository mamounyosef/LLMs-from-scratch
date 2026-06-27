import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers")
    .add_local_file("dataset.txt", "/root/dataset.txt")
    .add_local_python_source("llama3", "train")
)

app = modal.app("llama3-train")
volume = modal.Volume.from_name("llama3-checkpoints", create_if_missing=True)

@app.function(
    gpu="H200",
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.secret.from_name("huggingface")],
    cpu=4.0,
    memory= 16 * 1024
    timeout= 60 * 60 * 2
)
def train_remote():
    from train import run_training
    run_training()
    volume.commit()

@app.local_entrypoint()
def main():
    train_remote.remote()