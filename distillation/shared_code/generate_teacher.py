import json, os
from pathlib import Path

TEACHER = "Qwen/Qwen3.8-27B"
REASONING_EFFORT = "low"
MAX_NEW_TOKENS = 2048

PROMPTS_PATH = Path("/root/prompts_train.jsonl")
OUTPUT_DIR = Path("/outputs")

def load_prompts(limit=None):
    rows = [json.loads(line) for line in open(PROMPTS_PATH, encoding="utf-8")]
    if limit:
        return rows[:limit]
    return rows

def format_prompts(rows, tokenizer):
    texts = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=REASONING_EFFORT
        )
        texts.append(text)
    return texts

def generate(rows):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    texts = format_prompts(rows, tokenizer)

    model = LLM(
        model=TEACHER,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=4096
    )

    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS
    )
    outputs = model.generate(texts, sampling)

    results = []
    for row, output in zip(rows, outputs):
        completion = output.outputs[0]
        results.append({
            "prompt": row["prompt"],
            "domain": row["domain"],
            "response": completion.text,
            "num_tokens": len(completion.token_ids),
            "finish_reason": completion.finish_reason
        })
    return results

def save(results, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved: {len(results)} to {path}")

def summarize(results):
    import statistics, collections

    by_domain = collections.defaultdict(list)
    for row in results:
        by_domain[row["domain"]].append(row["num_tokens"])

    truncated = sum(1 for row in results if row["finish_reason"] == "length")
    total = 0
    for domain, values in by_domain.items():
        print(domain, "mean", int(statistics.mean(values)),
              "median", int(statistics.median(values)), "min", min(values), "max", max(values))
        total += sum(values)

    print("output tokens", total)
    print("mean per response", total // len(results))
    print("truncated at cap:", truncated, "/", len(results))

def run(limit=None):
    rows = load_prompts(limit)
    results = generate(rows)
    name = "teacher_test.jsonl" if limit else "teacher_full.jsonl"
    save(results, name)
    summarize(results)