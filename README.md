# embodied-agent-lab

把自己训练的小模型（MiniMind，约 64M 参数）包成一个 **OpenAI 兼容的 HTTP 服务**，
再用它做一个**会调用工具的 Agent**。

这是「方向二：大模型与智能体」三件套里缺的两件：
1. 自训模型 —— 已完成（`out/full_sft_768.pth` 导出成了 `scripts/minimind-3/`）
2. 部署服务 —— `server.py`
3. Agent 应用 —— `agent.py`

## 每个文件干什么

| 文件 | 作用 |
| --- | --- |
| `config.py` | 模型所在的文件夹路径，换电脑时只改这一行 |
| `check_model.py` | 冒烟测试：模型能不能加载、会不会正常回话 |
| `server.py` | 把模型包成 HTTP 服务，接口是 `/v1/chat/completions` |
| `agent.py` | Agent：模型自己决定调哪个工具（下一步写） |

## 怎么跑

```bash
# 1. 先确认模型能说话
python check_model.py

# 2. 起服务（占 8000 端口）
python server.py

# 3. 另开一个终端调它
curl -X POST http://127.0.0.1:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"

# 4. 跑 Agent（需要服务还开着）
python agent.py
```

Agent 的效果：

```
===== 提问：帮我算一下 256 乘以 37 等于多少 =====
[第1轮 模型] <tool_call>{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}</tool_call>
[执行工具] calculate_math({'expression': '256 * 37'}) -> {'result': 9472}
[第2轮 模型] 256 乘以 37 等于 9472。
```

模型自己决定调哪个工具、传什么参数，代码里只写了「有哪些工具」，没写「什么情况用哪个」。

返回结构和 OpenAI 一样，所以任何支持「自定义 base_url」的客户端都能直接接。

## 环境

- Python 3.11
- torch / transformers / fastapi / uvicorn
- 模型：<https://github.com/jingyaogong/minimind>
