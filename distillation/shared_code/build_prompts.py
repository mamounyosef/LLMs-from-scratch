import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json, random, re
from pathlib import Path
from datasets import load_dataset

seed = 42
PER_DOMAIN = 8000
OUT = Path(__file__).parent / "prompts_train.jsonl"
SPEC_BENCH_OUT = Path(__file__).parent / "spec_bench.jsonl"

SPEC_BENCH_URL = (
    "https://raw.githubusercontent.com/hemingkx/Spec-Bench/"
    "main/data/spec_bench/question.jsonl"    
)
random.seed(seed)

def norm(s: str):
    return re.sub(r"\s+", " ", s).strip().lower()

def spec_bench_prompts():
    import urllib.request
    with urllib.request.urlopen(SPEC_BENCH_URL) as f:
        rows = [json.loads(line) for line in f]
    with open(SPEC_BENCH_OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("saved", len(rows), "spec-bench rows ->", SPEC_BENCH_OUT)
    return {norm(turn) for row in rows for turn in row["turns"]}

def load_translation(count):
    dataset = load_dataset("wmt14", "de-en", split="train", streaming=True).shuffle(seed=seed, buffer_size=60_000)
    prompts = []
    for row in dataset:
        german = row["translation"]["de"]
        if 50 < len(german) < 500:
            prompts.append("Translate German to English: " + german)
        if len(prompts) >= count:
            break
    return prompts
def load_summarization(count):
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True).shuffle(seed=seed, buffer_size=60_000)
    prompts = []
    for row in dataset:
        article = row["article"]
        if 1000 < len(article) < 7000:
            prompts.append("Summarize: " + article)
        if len(prompts) >= count:
            break
    return prompts

def load_qa(count):
    dataset = load_dataset("google-research-datasets/nq_open", split="train").shuffle(seed=seed)
    return [row["question"] for row in dataset.select(range(count))]

def load_math(count):
    dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    count = min(count, len(dataset))
    return [row["question"] for row in dataset.select(range(count))]

def load_rag(count):
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True).shuffle(seed=seed, buffer_size=60_000)
    prompts = []
    for row in dataset:
        passages = row["passages"]["passage_text"]
        context = "\n".join(passages)
        if 1500 < len(context) < 4000:
            prompts.append(context + "\n" + row["query"])
        if len(prompts) >= count:
            break
    return prompts

def load_chat(count):
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed, buffer_size=60_000)
    prompts = []
    for row in dataset:
        first_message = row["messages"][0]["content"]
        if 20 < len(first_message) < 2000:
            prompts.append(first_message)
        if len(prompts) >= count:
            break
    return prompts

def main():
    blocked = spec_bench_prompts()
    print("spec-bench prompts to exclude:", len(blocked))

    loaders = {
        "translation": load_translation,
        "summarization": load_summarization,
        "qa": load_qa,
        "math": load_math,
        "rag": load_rag,
        "chat": load_chat
    }

    rows = []
    seen = set()

    for domain, loader in loaders.items():
        prompts = loader(PER_DOMAIN)
        kept = 0
        leaked = 0
        for prompt in prompts:
            key = norm(prompt)
            if key in blocked:
                leaked += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            rows.append({"prompt": prompt, "domain": domain})
            kept +=1
        print(domain, "kept", kept, "| leaked", leaked, "| raw", len(prompts))
    random.shuffle(rows)

    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("total", len(rows), ", ", OUT)

if __name__ == "__main__":
    main()
