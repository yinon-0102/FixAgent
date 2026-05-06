from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
import json
from schemas.request import ChatRequest, KnowledgeSearchRequest
from schemas.response import ChatResponse, KnowledgeSearchResponse, BaseResponse
from schemas.models import AgentMode
from services.llm_service import get_llm_service

# Agent 初始化（应用启动时创建，避免每次请求创建）


app = FastAPI(
    title="FixAgent AI Module",
    version="1.0.0",
    description="AI推理引擎：故障诊断、知识检索、作业指引"
)


"""对话，非流式响应"""
@app.post("/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        messages = [{"role": "user", "content": request.message}]
        result = await get_llm_service().chat(messages, stream=False)
        return ChatResponse(session_id=request.session_id, message=result["content"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""对话，流式响应"""
@app.post("/ai/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        messages = [{"role": "user", "content": request.message}]
        stream_iter = await get_llm_service().chat(messages, stream=True)

        yield f"data: {json.dumps({'event': 'session_id', 'data': {'session_id': request.session_id}})}\n\n"

        async for token in stream_iter:
            yield f"data: {json.dumps({'event': 'token', 'data': {'content': token}})}\n\n"

        yield f"data: {json.dumps({'event': 'done', 'data': {}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


"""检索"""
@app.post("/ai/retrieval", response_model=ChatResponse)
async def retrieval(request: ChatRequest) -> ChatResponse:
    #直接调用 RetrievalAgent，从向量库检索相关知识。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""诊断"""
@app.post("/ai/diagnosis", response_model=ChatResponse)
async def diagnosis(request: ChatRequest) -> ChatResponse:
    #直接调用 DiagnosisAgent，进行故障分析和原因推理。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""指引"""
@app.post("/ai/guidance", response_model=ChatResponse)
async def guidance(request: ChatRequest) -> ChatResponse:
    #直接调用 GuidanceAgent，生成标准化的维修作业步骤。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""完整流程"""
@app.post("/ai/pipeline", response_model=ChatResponse)
async def pipeline(request: ChatRequest) -> ChatResponse:
    #依次执行：检索 -> 诊断 -> 指引，返回综合分析结果。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""知识检索"""
@app.post("/ai/knowledge/search", response_model=KnowledgeSearchResponse)
async def knowledge_search(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    #直接调用向量检索服务，返回 TopK 相关片段。

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""全局异常处理"""
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return BaseResponse(
        success=False,
        message=str(exc),
        code=500
    )