import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from datetime import datetime

# 路径
MODEL_DIR = r"C:\Users\24437\Desktop\11111\实验室\minimind-master\scripts\minimind-3"

# 加载模型
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, dtype=torch.float32
).to("cuda").eval()

print("参数量", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# tools
def calculate_math(expression):
    return eval(expression, {"__builtins__": {}})

def get_current_time(timezone="Asia/Shanghai"):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

executors = {"calculate_math": calculate_math, "get_current_time": get_current_time}

# tools清单
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

# 测试集
tests = [
    ("帮我算一下 128 乘以 5", "calculate_math"),
    ("128乘以5是多少", "calculate_math"),
    ("256 乘 37 等于几", "calculate_math"),
    ("帮我算 100 除以 4", "calculate_math"),
    ("1加1等于几", "calculate_math"),
    ("现在几点了？", "get_current_time"),
    ("今天几号", "get_current_time"),
    ("现在是什么时间", "get_current_time"),
    ("你好", None),
    ("介绍一下你自己", None),
]

# 路由
def route(q):
    if re.search(r"\d", q) and re.search(r"乘|除|加|减|次方|平方|开方|立方|[+\-*/]", q):
        return "calculate_math"
    elif re.search(r"几|时间|日期|今天", q):
        return "get_current_time"
    return None

for q, expect in tests:
    got = route(q)
    print(f"{'对' if got == expect else '错'}  路由={got}  期望={expect}   {q}")


# 边界测试
edge = [
    ("帮我算一下 2 的 10 次方", "calculate_math"),
    ("根号 16 是多少", "calculate_math"),
    ("3 的平方", "calculate_math"),
    ("帮我算 2 的 3 次方加 1", "calculate_math"),
]

for q, expect in edge:
    got = route(q)
    print(f"{'对' if got == expect else '错'}  路由={got}  期望={expect}   {q}")


