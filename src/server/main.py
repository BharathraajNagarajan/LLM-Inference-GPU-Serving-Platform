"""Minimal FastAPI server wrapping the same model used in
src/baseline/naive_inference.py behind a single synchronous /generate
endpoint.

Still no vLLM, no batching, no concurrency handling — the model is loaded
once at startup and each request runs one synchronous generate() call.
This is the "baseline behind an API" step, bridging toward the vLLM
integration later.
"""

import threading
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 64

model_state: dict = {}

# threading.Lock, not asyncio.Lock: the /generate handler below is a sync
# `def`, which FastAPI runs in a worker thread pool (not on the asyncio
# event loop), so concurrent requests are already separate OS threads and
# an asyncio.Lock would not coordinate them at all. model.generate() is
# also a blocking call with no async-native equivalent, so making the
# endpoint `async def` would buy nothing here and would risk stalling the
# event loop if the lock were ever awaited incorrectly. This lock makes
# that thread-pool serialization explicit and enforced, rather than
# relying on incidental GIL/CUDA scheduling.
generate_lock = threading.Lock()


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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


def _generate_completion(messages: list[dict], max_new_tokens: int) -> dict:
    """Shared generation path for both /generate and /v1/chat/completions.

    Tokenizes `messages` via the chat template, runs model.generate()
    once under generate_lock, and returns token counts, completion text,
    timing, and an OpenAI-style finish_reason: "stop" if generation ended
    on the model's own EOS token, "length" if it was cut off by
    max_new_tokens first.
    """
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    device = model_state["device"]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    input_ids = encoded["input_ids"]
    num_input_tokens = input_ids.shape[-1]

    generate_start = time.perf_counter()
    with generate_lock, torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    generation_time = time.perf_counter() - generate_start

    new_tokens = output_ids[0, num_input_tokens:]
    num_output_tokens = new_tokens.shape[-1]
    completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    ended_on_eos = (
        num_output_tokens > 0
        and tokenizer.eos_token_id is not None
        and new_tokens[-1].item() == tokenizer.eos_token_id
    )
    finish_reason = "stop" if ended_on_eos else "length"

    return {
        "completion_text": completion_text,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "generation_time_s": generation_time,
        "finish_reason": finish_reason,
    }


# /generate is the original simple endpoint (single prompt in, single
# completion out). /v1/chat/completions is the OpenAI-compatible endpoint,
# added because vLLM's server is natively OpenAI-compatible — matching its
# request/response contract here makes the eventual naive-vs-vLLM
# benchmark comparison apples-to-apples instead of just a keyword match.
@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    result = _generate_completion(
        messages=[{"role": "user", "content": request.prompt}],
        max_new_tokens=request.max_new_tokens,
    )
    return GenerateResponse(
        completion=result["completion_text"],
        input_tokens=result["num_input_tokens"],
        output_tokens=result["num_output_tokens"],
        generation_time_s=result["generation_time_s"],
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    max_new_tokens = (
        request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    )
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = _generate_completion(messages=messages, max_new_tokens=max_new_tokens)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result["completion_text"]),
                finish_reason=result["finish_reason"],
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=result["num_input_tokens"],
            completion_tokens=result["num_output_tokens"],
            total_tokens=result["num_input_tokens"] + result["num_output_tokens"],
        ),
    )
