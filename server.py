# 把模型包成一个 OpenAI 兼容的 HTTP 服务
# 目标：任何"支持自定义 base_url 的客户端"（包括自己的 Agent）都能直接调
import re
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_DIR

app = FastAPI()

# 模型只在启动时加载一次，之后每个请求共用（否则每次请求都要重载 200MB）
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, dtype=torch.float32
).eval()


def clean(text: str) -> str:
    # 这个模型会先输出 <think> 思考段，演示时去掉，只留最终回答
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("</think>", "").replace("<think>", "").strip()


class ChatRequest(BaseModel):
    model: str = "minimind-64m"
    messages: list
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = False  # 先只做非流式，Agent 用不到流式
    tools: list = []  # 工具清单，模型靠它决定调哪个


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt = tok.apply_chat_template(
        req.messages, tools=req.tools or None, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 1e-5),
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )

    text = clean(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    # 这里刻意保持 OpenAI 的返回结构：换成本地模型，上层代码一行都不用改
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": out.shape[1] - inputs["input_ids"].shape[1],
            "total_tokens": out.shape[1],
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
