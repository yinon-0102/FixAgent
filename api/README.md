# API 模块

## 模块职责

FastAPI Web 服务入口，HTTP 接口定义、请求路由、参数校验。所有 AI 推理逻辑由 `agents/` 完成，业务数据由 Java 后端管理。

## 接口列表

| 接口 | 方法 | 描述 | 状态 |
|------|------|------|------|
| `/ai/chat` | POST | 对话接口（FixAgent → ReviewAgent） | **已实现** |
| `/ai/chat/stream` | POST | SSE 流式响应 | **已实现** |
| `/ai/knowledge/search` | POST | 直接调用 VectorService 检索 | **已实现** |
| `/ai/knowledge/import` | POST | 文档解析→向量化→入库管道 | **已实现** |
| `/ai/memory/consolidate` | POST | 记忆整理（function calling + 向量存储） | **已实现** |

### 已移除的接口

| 接口 | 移除原因 |
|------|---------|
| `/ai/retrieval` | FixAgent 统一处理，无需单独检索端点 |
| `/ai/diagnosis` | FixAgent 统一处理，无需单独诊断端点 |
| `/ai/guidance` | FixAgent 统一处理，无需单独指引端点 |
| `/ai/pipeline` | FixAgent 单次 ReAct 循环替代串行流水线 |

## 请求模型

`schemas/request.py` 中定义：

- `ChatRequest` — session_id / message(max_length=50000) / images / stream
- `KnowledgeImportRequest` — file_url / file_type / category / tags
- `KnowledgeSearchRequest` — query / top_k / category / tags
- `MemoryConsolidateRequest` — session_id / memoryMessages / memoryPreferenceVOList / memoryUnresolvedVOList

## 响应模型

`schemas/response.py` 中定义：

- `ChatResponse` — session_id / message / tools_used / react_trace / review_result / latency_ms
- `KnowledgeImportResponse` — file_name / total_pages / text_count / image_count / table_count / sections / extraction_summary
- `KnowledgeSearchResponse` — data(VectorSearchResult列表) / total / query_time_ms
- `MemoryConsolidateResponse` — session_id / summary(MemorySummary) / original_count / consolidated_at

## SSE 流式事件

```
data: {"event": "session_id", "data": {"session_id": "xxx"}}
data: {"event": "status",     "data": {"message": "正在检索知识库..."}}
data: {"event": "tool",       "data": {"name": "knowledge_retrieval", "arguments": {...}}}
data: {"event": "token",      "data": {"content": "根据"}}
data: {"event": "review",     "data": {"status": "approved"}}
data: {"event": "done",       "data": {"tools_used": [...], "latency_ms": 3200}}
data: {"event": "error",      "data": {"message": "..."}}
```

## 调用关系

```
api/main.py
    ├── schemas/request.py        — 请求模型
    ├── schemas/response.py       — 响应模型
    ├── agents/fix_agent.py       — 统一推理（单例，惰性创建）
    ├── agents/review_agent.py    — 输出审核（单例，惰性创建）
    ├── agents/memory_agent.py    — 记忆整理
    ├── services/vector_service.py    — 向量检索（knowledge/search）
    └── services/knowledge_service.py — 文档导入（knowledge/import）
```

## 与 Java 后端的交互

```
Java Backend                    FixAgent (Python)
  POST /ai/chat                     → FixAgent → ReviewAgent → ChatResponse
  POST /ai/chat/stream (SSE)       → FixAgent.stream → ReviewAgent → SSE 事件流
  POST /ai/knowledge/import        → KnowledgeService → KnowledgeImportResponse
  POST /ai/knowledge/search         → VectorService → KnowledgeSearchResponse
  POST /ai/memory/consolidate      → MemoryAgent → MemoryConsolidateResponse
```

## 错误处理

- Agent 执行失败（`metadata.status="error"`）→ API 层检测后 raise HTTPException(500)
- LLM 返回 content=null（tool_call 场景）→ `content or ""` 兜底
- JSON 解析失败 → 返回 error AgentOutput + warning 日志
- 请求参数校验失败 → FastAPI 自动返回 422
- 全局异常捕获 → JSONResponse(status_code=500) 返回给 Java

## 启动方式

```bash
# 开发环境（热重载）
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 生产环境
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 文件结构

```
api/
├── __init__.py
├── main.py          # FastAPI 入口，含 /ai/* 所有端点
└── asr_api.py       # 语音识别接口
```

## 注意事项

1. **日志级别**：生产环境将 `logging.basicConfig(level=logging.INFO)` 改为 `WARNING`
2. **Agent 惰性初始化**：应用启动时不加载 LLM，首次请求时才创建实例
3. **会话追踪**：`session_id` 由 Java 生成并传递，用于日志分片和链路追踪
4. **超时设置**：建议 HTTP 超时 > 60s（AI 推理耗时较长）
5. **SSE 协议**：流式接口推送 session_id / status / tool / token / review / done / error 事件
