# 冒烟测试：确认模型能加载、能回话
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_DIR

# 先只跑 CPU：模型只有约 54M 参数，CPU 足够快
# （换 GPU 要 cu128 版 PyTorch，否则 Blackwell 显卡会报 "no kernel image"）
torch.set_num_threads(8)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, dtype=torch.float32
).eval()

print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

msgs = [{"role": "user", "content": "用一句话介绍你自己"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# 不截断、不上采样，先看模型本身会说人话不
inputs = tok(prompt, return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )

reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("--- 回复 ---")
print(reply.strip())
