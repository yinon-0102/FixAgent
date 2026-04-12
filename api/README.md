# API 模块

## 模块职责

FastAPI Web 服务入口，负责 HTTP 接口、请求路由、参数校验。

> **重要**：本模块仅负责接收请求和返回响应，所有 AI 推理逻辑由 `agents/` 和 `chains/` 完成。

## 接口列表

| 接口 | 方法 | 描述 | Agent调用 |
|------|------|------|-----------|
| `/ai/chat` | POST | 对话接口（自动意图识别） | OrchestratorAgent |
| `/ai/retrieval` | POST | 纯检索接口 | RetrievalAgent |
| `/ai/diagnosis` | POST | 纯诊断接口 | DiagnosisAgent |
| `/ai/guidance` | POST | 纯指引接口 | GuidanceAgent |
| `/ai/pipeline` | POST | 完整流程接口 | All Agents |

## 请求模型

参见 `schemas/request.py`：

- `ChatRequest` - 对话请求
- `KnowledgeSearchRequest` - 知识检索请求
- `GraphQueryRequest` - 图谱查询请求
- `YoloDetectRequest` - YOLO检测请求
- `SamSegmentRequest` - SAM分割请求
- `ClipEmbedRequest` - CLIP向量化请求
- `DocumentParseRequest` - 文档解析请求

## 响应模型

参见 `schemas/response.py`：

- `ChatResponse` - 对话响应
- `KnowledgeSearchResponse` - 知识检索响应
- `GraphQueryResponse` - 图谱查询响应
- `YoloDetectResponse` - YOLO检测响应
- `SamSegmentResponse` - SAM分割响应
- `ClipEmbedResponse` - CLIP向量化响应
- `DocumentParseResponse` - 文档解析响应

## 依赖关系

```
api/main.py
    │
    ├── schemas/request.py      # 请求模型
    ├── schemas/response.py     # 响应模型
    ├── schemas/models.py       # 枚举和常量
    │
    ├── agents/orchestrator_agent.py
    ├── agents/retrieval_agent.py
    ├── agents/diagnosis_agent.py
    ├── agents/guidance_agent.py
    │
    └── services/llm_service.py     # 阿里云百炼
    └── services/vector_service.py  # Redis向量库
    └── services/graph_service.py    # Neo4j图数据库
```

## 开发注意事项

1. **请求参数校验**：使用 Pydantic 模型自动校验
2. **错误处理**：统一返回 `ErrorResponse` 格式
3. **日志追踪**：`session_id` 用于关联 Java 后端的会话
4. **流式输出**：`stream=True` 时使用 Server-Sent Events

## 环境变量

| 变量名 | 说明 | 来源 |
|--------|------|------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key | `.env` |
| `REDIS_HOST` | Redis 地址 | `.env` |
| `REDIS_PORT` | Redis 端口 | `.env` |
| `NEO4J_URI` | Neo4j 连接URI | `.env` |
| `NEO4J_USERNAME` | Neo4j 用户名 | `.env` |
| `NEO4J_PASSWORD` | Neo4j 密码 | `.env` |

## 启动方式

```bash
# 开发环境
uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

# 生产环境
uvicorn api.main:app --host 0.0.0.0 --port 8001 --workers 4
```

## 与 Java 后端的交互

```
Java Backend (8080)  ──── HTTP POST ────>  Python AI (8001)
                                              │
                                              ├── /ai/chat
                                              ├── /ai/retrieval
                                              ├── /ai/diagnosis
                                              ├── /ai/guidance
                                              └── /ai/pipeline
```
