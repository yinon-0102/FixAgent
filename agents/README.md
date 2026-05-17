# Agents 模块

## 模块概述

Agent 是系统的**核心智能组件**，采用 **FixAgent + ReviewAgent** 双层架构：

- **FixAgent（统一推理）**：持有全部工具的 ReAct Agent，在单次循环中自主决策工具调用顺序和次数
- **ReviewAgent（输出审核）**：纯 LLM 无工具，对 FixAgent 输出做最终质量校验（Reflexion 思想）
- **MemoryAgent（记忆整理）**：独立的 function calling Agent，由 `/ai/memory/consolidate` 端点单独调用

设计原则：移除意图路由层的额外延迟，让 LLM 在 ReAct 循环中自主决策。

## 架构图

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                FixAgent (统一推理)                      │
│                                                        │
│  系统提示词融合：知识检索 + 故障诊断 + 维修指引        │
│  ReAct 循环: Think → Action → Observation → Answer    │
│                                                        │
│  可用工具:                                             │
│  ├── knowledge_retrieval (知识库向量检索)              │
│  ├── graph_query_diagnosis_path (图谱诊断路径)        │
│  └── graph_search_devices (图谱设备搜索)              │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│              ReviewAgent (输出审核)                     │
│                                                        │
│  校验维度:                                             │
│  ├── 诊断依据充分性（有无凭空推测）                    │
│  ├── 安全提示完整性（高压/高温/化学品）                │
│  ├── 故障原因完整性（有无遗漏）                        │
│  └── 内容一致性（有无自相矛盾）                        │
│                                                        │
│  输出: approved → 原样返回 / revised → 修正后内容      │
└──────────────────────────────────────────────────────┘
```

## Agent 列表

| Agent | 文件 | 职责 | 执行模式 |
|-------|------|------|---------|
| `fix_agent` | fix_agent.py | 统一推理（检索+诊断+指引） | `run_with_react()` — ReAct 循环 |
| `review_agent` | review_agent.py | 输出质量审核 | `run()` — 单次 LLM 调用 |
| `memory_agent` | memory_agent.py | 记忆整理 | `run()` — function calling + Pydantic 校验 |

## Agent 基类

`BaseAgent` 定义统一执行流程和异常处理模板：

- `run()` — 标准模板方法
- `run_with_react()` — ReAct 入口，收集工具列表 → `chat_with_tools()` → 记录 `react_trace` 到 metadata
- `run_with_react_stream()` — ReAct 流式版本，yield SSE 事件
- `run_stream()` — 流式输出
- `run_with_context()` — 便捷方法，构造 `AgentInput` 后调用 `run()`

所有子 Agent 继承 `BaseAgent`，覆盖：

- `name` / `description` 属性
- `get_system_prompt()` — 返回角色定义提示词
- `get_tools()` — 返回可用工具列表（FixAgent 实现，ReviewAgent 不需要）
- `_build_messages()` — 自定义消息构建（MemoryAgent 覆盖）

异常处理：任意环节失败返回 `AgentOutput` 的友好提示 + `metadata.status="error"`，不抛出。

## 文件结构

```
agents/
├── __init__.py
├── base_agent.py          # Agent 基类，含 run()/run_with_react()/run_stream()
├── fix_agent.py           # 统一推理 ReAct Agent（持有全部工具）
├── review_agent.py        # 输出审核 Agent（纯 LLM，无工具）
└── memory_agent.py        # 记忆整理 function calling Agent
```

## 与其他模块的关系

```
agents/ (Agent层)
    ├── services/llm_service.py — chat() / chat_with_tools() / stream()
    ├── tools/ — FixAgent 的可用工具（通过 get_tools() 注入 ReAct 循环）
    ├── embeddings/text_embedding.py — MemoryAgent 向量化存储
    └── services/vector_service.py — MemoryAgent 事实检索
```

## ReAct Trace 可观测性

`run_with_react()` 执行后，推理轨迹写入 `AgentOutput.metadata`：

```json
{
  "execution_mode": "react",
  "react_trace": [
    {
      "iteration": 1,
      "action": "tool_call",
      "tool_calls": [
        {
          "name": "knowledge_retrieval",
          "arguments": {"query": "电动机轴承过热", "top_k": 5},
          "result_summary": "找到5条相关知识..."
        }
      ],
      "duration_ms": 1840
    },
    {
      "iteration": 2,
      "action": "tool_call",
      "tool_calls": [
        {
          "name": "graph_query_diagnosis_path",
          "arguments": {"keyword": "电动机", "fault_name": "轴承过热"},
          "result_summary": "找到3条诊断路径..."
        }
      ],
      "duration_ms": 920
    },
    {
      "iteration": 3,
      "action": "finish",
      "content_preview": "电动机轴承过热通常由以下原因引起...",
      "duration_ms": 1200
    }
  ],
  "react_iterations": 3,
  "review_result": "approved"
}
```

## 已删除的 Agent

| Agent | 文件 | 删除原因 |
|-------|------|---------|
| `orchestrator_agent` | orchestrator_agent.py | FixAgent 统一处理，不再需要调度层 |
| `retrieval_agent` | retrieval_agent.py | 检索能力由 FixAgent 内置 |
| `diagnosis_agent` | diagnosis_agent.py | 诊断能力由 FixAgent 内置 |
| `guidance_agent` | guidance_agent.py | 指引能力由 FixAgent 内置 |
| `intention_recognizer` | intention/recognizer.py | 移除意图识别，FixAgent 自主决策 |

## 注意事项

1. **ReAct 迭代上限**：默认 max_iterations=10，超出返回已有内容
2. **ReviewAgent 独立性**：不参与 ReAct 循环，仅在 FixAgent 完成后做一次校验
3. **MemoryAgent 独立性**：不通过 FixAgent 调度，直接由 `/ai/memory/consolidate` 调用
4. **流式输出**：FixAgent 支持 SSE 流式，ReviewAgent 在流式模式下返回 review 事件
