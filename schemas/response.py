"""
Schemas响应模型模块

定义所有API的响应数据模型，包括分页、错误处理等。
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field
from models import (
    BaseResponse, PaginationMeta,
    DetectionResult, VectorSearchResult, GraphNode, GraphRelation
)


# ==================== 对话相关 ====================

class ChatStreamEvent(BaseModel):
    """对话流式事件"""
    event: str = Field(..., description="事件类型: token/status/tool/done/error")
    data: Any = Field(..., description="事件数据")

    class Config:
        json_schema_extra = {
            "example": {
                "event": "token",
                "data": {"content": "维修"}
            }
        }


class ChatResponse(BaseResponse):
    """对话响应（非流式）"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="AI回复")
    intention: Optional[str] = Field(default=None, description="识别到的意图")
    tools_used: Optional[List[str]] = Field(default=None, description="使用的工具列表")
    latency_ms: Optional[int] = Field(default=None, description="响应延迟(ms)")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "操作成功",
                "code": 200,
                "session_id": "sess_abc123",
                "message": "电动机轴承过热可能由以下原因造成：1. 润滑不良...",
                "intention": "troubleshoot",
                "tools_used": ["knowledge_retrieval", "graph_query"],
                "latency_ms": 1500
            }
        }


# ==================== 知识库相关 ====================

class KnowledgeItem(BaseModel):
    """知识条目"""
    id: int
    title: str
    content: str
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    file_urls: Optional[List[str]] = None
    status: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class KnowledgeListResponse(BaseResponse):
    """知识列表响应"""
    data: List[KnowledgeItem]
    meta: PaginationMeta


class KnowledgeDetailResponse(BaseResponse):
    """知识详情响应"""
    data: KnowledgeItem


class KnowledgeSearchResponse(BaseResponse):
    """知识检索响应"""
    data: List[VectorSearchResult]
    total: int
    query_time_ms: int


# ==================== 案例相关 ====================

class CaseItem(BaseModel):
    """案例项"""
    id: int
    title: str
    description: str
    symptom: Optional[str] = None
    cause: Optional[str] = None
    solution: Optional[str] = None
    device_id: Optional[int] = None
    device_name: Optional[str] = None
    images: Optional[List[str]] = None
    status: str
    submitter_id: int
    submitter_name: Optional[str] = None
    reviewer_id: Optional[int] = None
    reviewer_name: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_comment: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class CaseListResponse(BaseResponse):
    """案例列表响应"""
    data: List[CaseItem]
    meta: PaginationMeta


class CaseDetailResponse(BaseResponse):
    """案例详情响应"""
    data: CaseItem


# ==================== 设备相关 ====================

class DeviceItem(BaseModel):
    """设备项"""
    id: int
    name: str
    model: Optional[str] = None
    category: Optional[str] = None
    manufacturer: Optional[str] = None
    specs: Optional[dict] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class DeviceListResponse(BaseResponse):
    """设备列表响应"""
    data: List[DeviceItem]
    meta: PaginationMeta


class DeviceDetailResponse(BaseResponse):
    """设备详情响应"""
    data: DeviceItem


# ==================== 图谱相关 ====================

class GraphQueryResponse(BaseResponse):
    """图谱查询响应"""
    nodes: List[GraphNode]
    relations: List[GraphRelation]
    query_time_ms: int


class GraphPathResponse(BaseResponse):
    """图谱路径查询响应"""
    paths: List[List[GraphNode]]
    total_paths: int


class GraphStatsResponse(BaseResponse):
    """图谱统计响应"""
    total_nodes: int
    total_relations: int
    node_types: dict
    relation_types: dict


# ==================== 工具调用相关 ====================

class YoloDetectResponse(BaseResponse):
    """YOLO检测响应"""
    image_url: str
    detections: List[DetectionResult]
    process_time_ms: int


class SamSegmentResponse(BaseResponse):
    """SAM分割响应"""
    image_url: str
    masks: List[dict]
    labels: List[str]
    process_time_ms: int


class ClipEmbedResponse(BaseResponse):
    """CLIP向量化响应"""
    embedding: List[float]
    dimension: int
    model: str


class DocumentParseResponse(BaseResponse):
    """文档解析响应"""
    file_name: str
    total_pages: int
    pages: List[dict]
    tables: List[dict]
    images: List[str]
    process_time_ms: int


# ==================== 任务相关 ====================

class TaskStatusResponse(BaseResponse):
    """任务状态响应"""
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str