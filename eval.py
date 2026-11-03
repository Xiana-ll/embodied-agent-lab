"""工具选择评测：同一批问题，对比「模型自主判断」与「规则路由」。

运行：python eval.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from agent import parse_calls, route, tools
from config import MODEL_DIR

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, dtype=torch.float32
).to("cuda").eval()

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

# 模型自主判断
print("=== 模型自主 ===")
right = 0
for q, expect in tests:
    messages = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    calls = parse_calls(reply)
    got = calls[0][0] if calls else None
    if got == expect:
        right += 1
    print(f"{'对' if got == expect else '错'}  期望={expect}  实际={got}   {q}")
print(f"正确率 {right}/{len(tests)} = {right/len(tests)*100:.0f}%")

# 规则路由
print("\n=== 规则路由 ===")
right = 0
for q, expect in tests:
    got = route(q)
    if got == expect:
        right += 1
    print(f"{'对' if got == expect else '错'}  期望={expect}  路由={got}   {q}")
print(f"正确率 {right}/{len(tests)} = {right/len(tests)*100:.0f}%")
