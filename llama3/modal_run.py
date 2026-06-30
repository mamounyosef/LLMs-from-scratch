import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers")
    .add_local_file("../dataset.txt", "/root/dataset.txt")
    .add_local_python_source("llama3", "llama3_train")
)

app = modal.App("llama3-train")
volume = modal.Volume.from_name("llama3-checkpoints", create_if_missing=True)

@app.function(
    gpu="B200",
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=4.0,
    memory= 16 * 1024,
    timeout= 60 * 60 * 2
)
def train_remote():
    from llama3_train import run_training
    run_training()
    volume.commit()

@app.local_entrypoint()
def main():
    train_remote.remote()