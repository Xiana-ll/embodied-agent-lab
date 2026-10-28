# 把模型包成一个 OpenAI 兼容的 HTTP 服务
# 目标：任何"支持自定义 base_url 的客户端"（包括自己的 Agent）都能直接调
import json
import re
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from agent import executors, parse_calls
from agent import tools as TOOLS
from config import MODEL_DIR

ROOT = Path(__file__).parent

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


def generate(messages: list, use_tools: bool, temperature: float = 0.7, max_tokens: int = 256):
    """跑一次模型。聊天和 Agent 都走这一个函数，避免两处逻辑不一致。"""
    prompt = tok.apply_chat_template(
        messages,
        tools=TOOLS if use_tools else None,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )

    text = clean(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return text, inputs["input_ids"].shape[1], out.shape[1]


class ChatRequest(BaseModel):
    model: str = "minimind-64m"
    messages: list
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = False  # 先只做非流式，Agent 用不到流式
    tools: list = []  # 工具清单，模型靠它决定调哪个


class AgentRequest(BaseModel):
    question: str
    max_rounds: int = 3


@app.get("/")
def index():
    # 网页聊天界面：和 API 用同一套后端，打开就是 http://127.0.0.1:8000/
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    text, prompt_tokens, total_tokens = generate(
        req.messages, use_tools=bool(req.tools), temperature=req.temperature, max_tokens=req.max_tokens
    )

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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_tokens - prompt_tokens,
            "total_tokens": total_tokens,
        },
    }


@app.post("/v1/agent")
def agent(req: AgentRequest):
    """在服务内部跑完整的工具调用循环，并把每一步的轨迹返回给网页显示。

    和 agent.py 的区别：agent.py 是通过 HTTP 请求自己，这里直接调本地模型。
    逻辑（parse_calls / executors）是同一份代码，没有复制。
    """
    messages = [{"role": "user", "content": req.question}]
    trace = []

    for _ in range(req.max_rounds):
        reply, _, _ = generate(messages, use_tools=True)
        calls = parse_calls(reply)
        if not calls:
            return {"answer": reply, "trace": trace}

        messages.append({"role": "assistant", "content": reply})
        for name, args in calls:
            try:
                result = executors[name](**args)
            except Exception as e:  # 参数是模型编的，可能算不出来
                result = {"error": str(e)}
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    return {"answer": reply, "trace": trace}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
