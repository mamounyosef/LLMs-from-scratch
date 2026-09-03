import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "transformers", "hugging_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models"})
    .add_local_file("prompts_train.jsonl", "/root/prompts_train.jsonl")
    .add_local_python_source("generate_teacher")
)

app = modal.App("distill-generate")
outputs = modal.Volume.from_name("distill-outputs", create_if_missing=True)
models = modal.Volume.from_name("distill-models", create_if_missing=True)