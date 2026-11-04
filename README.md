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
| `agent.py` | 工具清单 + 执行器 + 路由；直接运行会加载模型跑一遍完整流程 |
| `server.py` | 把模型包成 HTTP 服务：`/v1/chat/completions`、`/v1/agent`、网页 `/` |
| `web/index.html` | 聊天页面，纯 HTML+JS，调上面两个接口 |
| `start_demo.bat` | 一键启动 |
| `eval.py` | 评测：10 条测试集，对比「模型自主判断」与「规则路由」 |
| `logs/debug.md` | 踩坑记录：现象 → 原因 → 解决 |

## 两条实现路线

同一件事做了两遍，用途不同：

| | 服务版（实际用的） | 单文件版（对照用） |
| --- | --- | --- |
| 入口 | `start_demo.bat` → `server.py` | `python agent.py` |
| 模型加载 | 启动时一次，所有请求共用 | 每次运行加载一次 |
| 工具选择 | **规则路由**，模型不参与 | 模型自主（演示它选不对） |
| 回答生成 | 模板拼接，数字直接来自工具 | 模型自己说 |

服务版是能用的那个：`route()` 决定调哪个工具，模板把结果拼成一句话。
单文件版留着做对照——它就是「让模型自己选」的原始形态，实测选不对。

## 架构

```
浏览器 ──► server.py
              ├─ route() 决定用哪个工具
              ├─ 执行 calculate_math / get_current_time
              └─ 模板拼出答案

           MiniMind 63.9M ← 只在开放问题、闲聊时出场
```

- 有确定答案的问题（算数、查时间）**不经过模型**，避免它把 110 说成 212；
- 模型只处理 `route()` 返回 `None` 的开放问题；
- 工具定义和执行器只在 `agent.py` 里写一份，`server.py`、`eval.py` 都 import 它。

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

## 实际效果

```
问: 55乘2
  trace: [{'tool': 'calculate_math', 'args': {'expression': '55*2'}, 'result': 110}]
  答: 55*2 = 110

问: 5的20次方是多少
  trace: [{'tool': 'calculate_math', 'args': {'expression': '5**20'}, 'result': 95367431640625}]
  答: 5**20 = 95367431640625
```

## 已知边界

- **模型自己选工具不成立**：它不读工具清单，只复述训练数据里的模板。10 条测试里 8 条输出同一个工具名。
- **模型不复述工具结果**：算对了 110，它还是会说成 212。所以确定性问题改由代码拼答案。
- 事实性幻觉明显：知道「怎么回答」，不知道「回答什么」。
- 只做非流式输出。

## 环境

- Python 3.11 / torch 2.11.0+cu128 / transformers 4.57.6 / fastapi / uvicorn
- 模型：`C:\Users\24437\Desktop\11111\minimind-3`（122 MB，不进仓库）
- 模型来源：<https://github.com/jingyaogong/minimind>
