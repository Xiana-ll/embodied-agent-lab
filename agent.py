"""工具层 + 本地运行入口。

其他文件怎么用它：
    server.py   from agent import tools, executors, parse_calls
    eval.py     from agent import tools, route, parse_calls

直接运行（python agent.py）会加载模型，走一遍完整流程。
"""
import json
import re
from datetime import datetime

# 工具实现
def calculate_math(expression):
    return eval(expression, {"__builtins__": {}})

def get_current_time(timezone="Asia/Shanghai"):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

executors = {"calculate_math": calculate_math, "get_current_time": get_current_time}

# 工具清单
tools = [
    {"type": "function", "function": {
        "name": "calculate_math",
        "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string", "description": "数学表达式，如 23+456、2**10、sqrt(144)"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "获取当前日期和时间，支持指定时区",
        "parameters": {"type": "object",
            "properties": {"timezone": {"type": "string", "description": "时区名，如 Asia/Shanghai", "default": "Asia/Shanghai"}},
            "required": []}}},
]

def parse_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        calls.append((obj.get("name"), obj.get("arguments", {})))
    return calls

def route(q):
    if re.search(r"\d", q) and re.search(r"乘|除|加|减|次方|平方|开方|立方|[+\-*/]", q):
        return "calculate_math"
    elif re.search(r"几|时间|日期|今天", q):
        return "get_current_time"
    return None

def extract_expr(q):
    for cn, en in [("乘以", "*"), ("除以", "/"), ("加上", "+"), ("减去", "-"),
                   ("乘", "*"), ("除", "/"), ("加", "+"), ("减", "-")]:
        q = q.replace(cn, en)
    m = re.search(r"(\d+)\s*的\s*(\d+)\s*次方", q)
    if m:
        return f"{m.group(1)}**{m.group(2)}"
    return re.sub(r"[^\d+\-*/().\s]", " ", q).strip()


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from config import MODEL_DIR

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=torch.float32
    ).to("cuda").eval()

    messages = [{"role": "user", "content": "帮我算一下 128 乘以 5"}]

    prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("模型输出:", reply.strip())

    calls = parse_calls(reply)
    if calls:
        name, args = calls[0]
        result = executors[name](**args)
        print("工具结果:", result)

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "tool", "content": str(result)})

        prompt2 = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        inputs2 = tok(prompt2, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out2 = model.generate(**inputs2, max_new_tokens=64, do_sample=False,
                                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        print("最终回答:", tok.decode(out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    else:
        print("模型没调工具")
