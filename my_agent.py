
# 对话
messages = [{"role": "user", "content": "1+2+3+4等于几"}]
q = messages[0]["content"]

# 路由
if re.search(r"\d", q) and re.search(r"乘|除|加|减|[+\-*/]", q):
    expr = q
    for cn, en in [("乘以", "*"), ("除以", "/"), ("加上", "+"), ("减去", "-")]:
        expr = expr.replace(cn, en)
    expr = re.sub(r"[^\d+\-*/().\s]", " ", expr).strip()
    name, args = "calculate_math", {"expression": expr}
elif re.search(r"几|时间|日期|今天", q):
    name, args = "get_current_time", {}
else:
    name, args = None, {}

# 执行
if name:
    result = executors[name](**args)
    print("路由到:", name, args)
    print("工具结果:", result)

    # 把这一轮记进对话
    call_text = f'<tool_call>{{"name": "{name}", "arguments": {json.dumps(args, ensure_ascii=False)}}}</tool_call>'
    messages.append({"role": "assistant", "content": call_text})
    messages.append({"role": "tool", "content": str(result)})

    # 再让模型说一次
    prompt2 = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    inputs2 = tok(prompt2, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out2 = model.generate(**inputs2, max_new_tokens=64, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)

    print(tok.decode(out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True))
else:
    print("不需要工具")
