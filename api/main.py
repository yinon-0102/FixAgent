# api/main.py
"""
FastAPI Web 服务入口

职责：
- HTTP 接口定义
- 请求参数校验（依赖 schemas/）
- 调用 agents/ 执行 AI 推理
- 返回结构化响应

边界：
- 仅负责 AI 推理，不碰业务数据
- 会话历史由 Java 后端管理（Redis ChatMemory）
- 错误码返回给 Java，由 Java 展示友好提示
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List

from schemas.request import ChatRequest, KnowledgeSearchRequest
from schemas.response import ChatResponse, KnowledgeSearchResponse, BaseResponse
from schemas.models import AgentMode

# Agent 初始化（应用启动时创建，避免每次请求创建）
orchestrator_agent = OrchestratorAgent()
retrieval_agent = RetrievalAgent()
diagnosis_agent = DiagnosisAgent()
guidance_agent = GuidanceAgent()

app = FastAPI(
    title="FixAgent AI Module",
    version="1.0.0",
    description="AI推理引擎：故障诊断、知识检索、作业指引"
)


# ==================== 对话相关接口 ====================

@app.post("/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    对话接口（自动意图识别）

    流程：
    1. Java 后端组装 AIContext（包含会话历史）
    2. 调用本接口，传入用户消息和上下文
    3. OrchestratorAgent 识别意图并调度
    4. 返回 AI 推理结果

    参数：
        session_id: Java后端传递，用于日志追踪
        message: 用户消息
        mode: 运行模式（CHAT/RETRIEVAL/DIAGNOSIS/GUIDANCE/FULL）
        images: 图片URL列表
        stream: 是否流式输出
    """
    try:
        result = await orchestrator_agent.run(
            session_id=request.session_id,
            message=request.message,
            mode=request.mode,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention=result.intention,
            tools_used=result.tools_used,
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/retrieval", response_model=ChatResponse)
async def retrieval(request: ChatRequest) -> ChatResponse:
    """
    纯检索接口

    直接调用 RetrievalAgent，从向量库检索相关知识。
    """
    try:
        result = await retrieval_agent.run(
            session_id=request.session_id,
            message=request.message,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention="query_knowledge",
            tools_used=["knowledge_retrieval"],
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/diagnosis", response_model=ChatResponse)
async def diagnosis(request: ChatRequest) -> ChatResponse:
    """
    纯诊断接口

    直接调用 DiagnosisAgent，进行故障分析和原因推理。
    """
    try:
        result = await diagnosis_agent.run(
            session_id=request.session_id,
            message=request.message,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention="troubleshoot",
            tools_used=["knowledge_retrieval", "graph_query"],
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/guidance", response_model=ChatResponse)
async def guidance(request: ChatRequest) -> ChatResponse:
    """
    纯指引接口

    直接调用 GuidanceAgent，生成标准化的维修作业步骤。
    """
    try:
        result = await guidance_agent.run(
            session_id=request.session_id,
            message=request.message,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention="seek_guidance",
            tools_used=["knowledge_retrieval"],
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/pipeline", response_model=ChatResponse)
async def pipeline(request: ChatRequest) -> ChatResponse:
    """
    完整流程接口

    依次执行：检索 -> 诊断 -> 指引，返回综合分析结果。
    """
    try:
        result = await orchestrator_agent.run_full_pipeline(
            session_id=request.session_id,
            message=request.message,
            mode=AgentMode.FULL,
            images=request.images or []
        )

        return ChatResponse(
            session_id=request.session_id,
            message=result.message,
            intention="full_pipeline",
            tools_used=["knowledge_retrieval", "graph_query"],
            latency_ms=result.latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 工具类接口 ====================

@app.post("/ai/knowledge/search", response_model=KnowledgeSearchResponse)
async def knowledge_search(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """
    知识检索接口

    直接调用向量检索服务，返回 TopK 相关片段。
    """
    try:
        result = await vector_service.search(
            query=request.query,
            images=request.images or [],
            top_k=request.top_k,
            category=request.category
        )

        return KnowledgeSearchResponse(
            data=result.results,
            total=result.total,
            query_time_ms=result.query_time_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 错误处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    return BaseResponse(
        success=False,
        message=str(exc),
        code=500
    )