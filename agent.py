"""最小 Agent：模型自己决定「要不要调工具、调哪个」。

流程只有三步，循环最多 3 轮：
  1. 把问题 + 工具清单发给模型
  2. 模型如果回 <tool_call>，就真的去执行那个工具
  3. 把执行结果再喂回模型，让它给最终回答
"""
import json
import math
import re
from datetime import datetime

import requests

url = "http://127.0.0.1:8000/v1/chat/completions"

# 工具定义要和模型训练时见过的一致，否则它不认得这个名字
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式，如 23+456、2**10、sqrt(144)"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间，支持指定时区",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "时区名，如 Asia/Shanghai", "default": "Asia/Shanghai"}},
                "required": [],
            },
        },
    },
]


def calculate_math(expression: str) -> dict:
    # 换成本地符号，再用 eval 算
    # 传了 {"__builtins__": {}}，模型就算吐出别的代码也执行不了
    expr = str(expression).replace("^", "**").replace("×", "*").replace("÷", "/")
    return {"result": eval(expr, {"__builtins__": {}}, {"sqrt": math.sqrt})}


def get_current_time(timezone: str = "Asia/Shanghai") -> dict:
    return {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": timezone}


# 名字 → 真正的函数。模型只吐出名字，具体执行在这里
executors = {"calculate_math": calculate_math, "get_current_time": get_current_time}


def parse_calls(text: str):
    """从模型输出里抠出所有 <tool_call>{"name":..,"arguments":..}</tool_call>

    模型可能一次要调好几个工具，所以是列表而不是单个。
    """
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue  # 吐了半截 JSON，跳过这一条
        name, args = obj.get("name"), obj.get("arguments", {})
        if name not in executors:
            continue
        if isinstance(args, str):  # 有的模型把 arguments 也编成字符串
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        calls.append((name, args))
    return calls


def ask(messages):
    r = requests.post(url, json={"messages": messages, "tools": tools, "temperature": 0.7, "max_tokens": 256}, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run(question: str):
    messages = [{"role": "user", "content": question}]
    print(f"\n===== 提问：{question} =====")

    for step in range(3):
        reply = ask(messages)
        print(f"[第{step + 1}轮 模型] {reply}")

        calls = parse_calls(reply)
        if not calls:  # 没调工具，说明它直接答了
            print("[结束] 模型直接回答，没调工具")
            return reply

        # 把「模型要调工具」和「每个工具的结果」都写回对话，模型下一轮才看得见
        messages.append({"role": "assistant", "content": reply})
        for name, args in calls:
            try:
                result = executors[name](**args)
            except Exception as e:  # 参数是模型编的，可能算不出来
                result = {"error": str(e)}
            print(f"[执行工具] {name}({args}) -> {result}")
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    print("[结束] 达到最大轮数")
    return reply


if __name__ == "__main__":
    run("帮我算一下 256 乘以 37 等于多少")
    run("现在几点了？")
    run("你好，介绍一下你自己")
