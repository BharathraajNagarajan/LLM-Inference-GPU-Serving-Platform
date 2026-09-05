"""Minimal FastAPI server wrapping the same model used in
src/baseline/naive_inference.py behind a single synchronous /generate
endpoint.

Still no vLLM, no batching, no concurrency handling — the model is loaded
once at startup and each request runs one synchronous generate() call.
This is the "baseline behind an API" step, bridging toward the vLLM
integration later.
"""

import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 64

model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    print(f"Loading {MODEL_NAME} on device={device}, dtype={dtype} ...")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model = model.to(device)
    model.eval()
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.4f}s")

    model_state["tokenizer"] = tokenizer
    model_state["model"] = model
    model_state["device"] = device

    yield

    model_state.clear()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


class GenerateResponse(BaseModel):
    completion: str
    input_tokens: int
    output_tokens: int
    generation_time_s: float


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    device = model_state["device"]

    messages = [{"role": "user", "content": request.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    input_ids = encoded["input_ids"]
    num_input_tokens = input_ids.shape[-1]

    generate_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=encoded["attention_mask"],
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    generation_time = time.perf_counter() - generate_start

    new_tokens = output_ids[0, num_input_tokens:]
    num_output_tokens = new_tokens.shape[-1]
    completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return GenerateResponse(
        completion=completion_text,
        input_tokens=num_input_tokens,
        output_tokens=num_output_tokens,
        generation_time_s=generation_time,
    )
