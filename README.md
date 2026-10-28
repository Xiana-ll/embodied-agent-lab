# embodied-agent-lab

把自己训练的小模型（MiniMind-3，实测 **63.91M** 参数）包成 **OpenAI 兼容的 HTTP 服务**，
再用它做一个**会调用工具的 Agent**，并配一个网页界面。

这是「方向二：大模型与智能体」三件套：

| # | 交付项 | 在哪 |
| --- | --- | --- |
| 1 | 自训模型 | `out/full_sft_768.pth` → 导出为 `scripts/minimind-3/`（存本地，不进仓库） |
| 2 | 部署服务 | `server.py` |
| 3 | Agent 应用 | `agent.py`（命令行）+ `/v1/agent`（网页） |

## 一键运行

双击 **`start_demo.bat`**：启动服务 → 10 秒后自动打开 <http://127.0.0.1:8000/>。

网页里勾上「启用工具（Agent）」，同一个输入框就能对比两种模式——

| 问「256 乘以 37」 | 不勾工具 | 勾上工具 |
| --- | --- | --- |
| 结果 | 胡编一段话 | 调 `calculate_math`，答 **9472** |

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `config.py` | 模型所在的文件夹路径，换电脑时只改这一行 |
| `check_model.py` | 冒烟测试：模型能不能加载、会不会正常回话 |
| `agent.py` | 工具定义 + 工具执行 + 调用循环（也是唯一一份「工具清单」） |
| `server.py` | 把模型包成 HTTP 服务：`/v1/chat/completions`、`/v1/agent`、网页 `/` |
| `web/index.html` | 聊天页面，纯 HTML+JS，调上面两个接口 |
| `start_demo.bat` | 一键启动 |
| `logs/debug.md` | 踩坑记录：现象 → 原因 → 解决 |

## 架构

```
浏览器 ──┐
         ├─► server.py ──► MiniMind 63.9M（本地权重）
agent.py ┘        │
                  └─ 工具：calculate_math / get_current_time
```

- `server.py` 启动时加载一次模型，所有请求共用；
- 网页和 `agent.py` 走的是**同一套接口**，`agent.py` 只是用 HTTP 请求自己；
- 工具循环的逻辑（`parse_calls` / `executors`）只写了一份，在 `agent.py` 里，`server.py` 直接 import。

## 手工运行

```bash
# 1. 先确认模型能说话
python check_model.py

# 2. 起服务（占 8000 端口）
python server.py

# 3. 打开网页
#    http://127.0.0.1:8000/

# 4. 或者用命令行调 API
curl -X POST http://127.0.0.1:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"

# 5. 命令行跑 Agent 循环（需要服务还开着）
python agent.py
```

## Agent 的效果

```
===== 提问：帮我算一下 256 乘以 37 等于多少 =====
[第1轮 模型] <tool_call>{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}</tool_call>
[执行工具] calculate_math({'expression': '256 * 37'}) -> {'result': 9472}
[第2轮 模型] 256 乘以 37 等于 9472。
```

设计上，模型自己决定调哪个工具、传什么参数——代码里只写了「有哪些工具」，**没写「什么情况用哪个」**。
但实测这个「决定」并未真正成立，见下方「已知边界」。

## 已知边界

- **工具选择并不成立**：模型没有读取工具清单，而是在复述训练数据里的模板。把工具裁剪到只剩计算器，它仍会输出清单之外的工具名 `get_current_time`。详见 [`logs/debug.md` 第 3.2 节](logs/debug.md)。
- **演示句式**：`帮我算一下 128 乘以 5` 能正确触发计算器；`128乘以5是多少` 会失败——后者可作为对照，说明它靠的是句式匹配而不是理解。
- 事实性幻觉明显：知道「怎么回答」，不知道「回答什么」；
- 只做非流式输出。

## 环境

- Python 3.11 / torch 2.13.0+cu126 / transformers 4.57.6 / fastapi / uvicorn
- 模型来源：<https://github.com/jingyaogong/minimind>
