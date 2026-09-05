"""Minimal, non-vLLM baseline inference script.

Loads Qwen/Qwen2.5-0.5B-Instruct via transformers and runs a single
synchronous prompt-in/completion-out generation, timing each stage.

No batching, no concurrency, no HTTP server, no vLLM. This establishes a
measured baseline before introducing serving-system complexity.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "What is the capital of France?"
MAX_NEW_TOKENS = 64


def main() -> None:
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
        print(
            "WARNING: CUDA is not available in this environment. "
            "Falling back to CPU. Generation will be significantly slower "
            "and torch.cuda memory stats will not be reported."
        )

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Model: {MODEL_NAME}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model = model.to(device)
    model.eval()
    load_end = time.perf_counter()
    load_time = load_end - load_start

    messages = [{"role": "user", "content": PROMPT}]

    tokenize_start = time.perf_counter()
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    tokenize_end = time.perf_counter()
    tokenize_time = tokenize_end - tokenize_start

    input_ids = encoded["input_ids"]
    num_input_tokens = input_ids.shape[-1]

    generate_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=encoded["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    generate_end = time.perf_counter()
    generate_time = generate_end - generate_start

    new_tokens = output_ids[0, num_input_tokens:]
    num_output_tokens = new_tokens.shape[-1]
    completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print()
    print("--- Prompt ---")
    print(PROMPT)
    print()
    print("--- Completion ---")
    print(completion_text)
    print()
    print("--- Timing / Stats ---")
    print(f"Model load time:   {load_time:.4f} s")
    print(f"Tokenization time: {tokenize_time:.4f} s")
    print(f"Generation time:   {generate_time:.4f} s")
    print(f"Input tokens:      {num_input_tokens}")
    print(f"Output tokens:     {num_output_tokens}")

    if device == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_mb = peak_mem_bytes / (1024 ** 2)
        print(f"Peak GPU memory:   {peak_mem_mb:.2f} MiB")
    else:
        print("Peak GPU memory:   N/A (ran on CPU)")


if __name__ == "__main__":
    main()
