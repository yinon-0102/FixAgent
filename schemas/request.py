"""
Schemas请求模型模块

定义所有API的请求数据模型，包括参数验证和默认值处理。
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
from models import AgentMode, CaseStatus


# ==================== 对话相关 ====================

class ChatRequest(BaseModel):
    """对话请求"""
    session_id: str = Field(..., description="会话ID，用于追踪对话历史")
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    mode: AgentMode = Field(default=AgentMode.CHAT, description="运行模式")
    images: Optional[List[str]] = Field(default=None, description="图片URL列表")
    stream: bool = Field(default=True, description="是否启用流式输出")

    @validator('images')
    def validate_images(cls, v):
        if v and len(v) > 10:
            raise ValueError("最多支持10张图片")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "message": "电动机轴承过热是什么原因？",
                "mode": "diagnosis",
                "images": ["https://example.com/fault1.jpg"],
                "stream": True
            }
        }


# ==================== 知识库相关 ====================

class KnowledgeCreateRequest(BaseModel):
    """创建知识条目请求"""
    title: str = Field(..., min_length=1, max_length=255, description="标题")
    content: str = Field(..., min_length=1, description="内容")
    category: Optional[str] = Field(default=None, max_length=50, description="分类")
    tags: Optional[List[str]] = Field(default=None, description="标签列表")
    file_urls: Optional[List[str]] = Field(default=None, description="关联文件URLs")


class KnowledgeUpdateRequest(BaseModel):
    """更新知识条目请求"""
    title: Optional[str] = Field(default=None, max_length=255, description="标题")
    content: Optional[str] = Field(default=None, description="内容")
    category: Optional[str] = Field(default=None, max_length=50, description="分类")
    tags: Optional[List[str]] = Field(default=None, description="标签列表")
    status: Optional[str] = Field(default=None, description="状态")


class KnowledgeSearchRequest(BaseModel):
    """知识检索请求"""
    query: str = Field(..., min_length=1, description="查询文本")
    images: Optional[List[str]] = Field(default=None, description="查询图片")
    top_k: int = Field(default=10, ge=1, le=50, description="返回数量")
    category: Optional[str] = Field(default=None, description="分类过滤")
    tags: Optional[List[str]] = Field(default=None, description="标签过滤")


class KnowledgeUploadRequest(BaseModel):
    """知识上传请求"""
    title: str = Field(..., description="文档标题")
    file_name: str = Field(..., description="文件名")
    file_url: str = Field(..., description="文件URL")
    category: Optional[str] = Field(default=None, description="分类")


# ==================== 案例相关 ====================

class CaseCreateRequest(BaseModel):
    """创建案例请求"""
    title: str = Field(..., min_length=1, max_length=255, description="案例标题")
    description: str = Field(..., description="故障描述")
    symptom: Optional[str] = Field(default=None, description="故障现象")
    cause: Optional[str] = Field(default=None, description="故障原因")
    solution: Optional[str] = Field(default=None, description="解决方案")
    device_id: Optional[int] = Field(default=None, description="关联设备ID")
    images: Optional[List[str]] = Field(default=None, description="故障图片URLs")


class CaseUpdateRequest(BaseModel):
    """更新案例请求"""
    title: Optional[str] = Field(default=None, max_length=255, description="案例标题")
    description: Optional[str] = Field(default=None, description="故障描述")
    symptom: Optional[str] = Field(default=None, description="故障现象")
    cause: Optional[str] = Field(default=None, description="故障原因")
    solution: Optional[str] = Field(default=None, description="解决方案")
    device_id: Optional[int] = Field(default=None, description="关联设备ID")
    images: Optional[List[str]] = Field(default=None, description="故障图片URLs")


class CaseSubmitRequest(BaseModel):
    """提交案例审核请求"""
    case_id: int = Field(..., description="案例ID")
    submitter_comment: Optional[str] = Field(default=None, description="提交说明")


class CaseReviewRequest(BaseModel):
    """审核案例请求"""
    case_id: int = Field(..., description="案例ID")
    status: CaseStatus = Field(..., description="审核状态")
    review_comment: Optional[str] = Field(default=None, description="审核意见")


# ==================== 设备相关 ====================

class DeviceCreateRequest(BaseModel):
    """创建设备请求"""
    name: str = Field(..., min_length=1, max_length=100, description="设备名称")
    model: Optional[str] = Field(default=None, max_length=100, description="设备型号")
    category: Optional[str] = Field(default=None, max_length=50, description="设备类别")
    manufacturer: Optional[str] = Field(default=None, max_length=100, description="制造商")
    specs: Optional[dict] = Field(default=None, description="规格参数")


class DeviceUpdateRequest(BaseModel):
    """更新设备请求"""
    name: Optional[str] = Field(default=None, max_length=100, description="设备名称")
    model: Optional[str] = Field(default=None, max_length=100, description="设备型号")
    category: Optional[str] = Field(default=None, max_length=50, description="设备类别")
    manufacturer: Optional[str] = Field(default=None, max_length=100, description="制造商")
    specs: Optional[dict] = Field(default=None, description="规格参数")


# ==================== 图谱相关 ====================

class GraphQueryRequest(BaseModel):
    """图谱查询请求"""
    entity_name: Optional[str] = Field(default=None, description="实体名称")
    entity_type: Optional[str] = Field(default=None, description="实体类型")
    relation_type: Optional[str] = Field(default=None, description="关系类型")
    depth: int = Field(default=1, ge=1, le=3, description="查询深度")


class GraphPathRequest(BaseModel):
    """图谱路径查询请求"""
    source_name: str = Field(..., description="起点实体名称")
    target_name: str = Field(..., description="终点实体名称")
    max_hops: int = Field(default=3, ge=1, le=5, description="最大跳数")


# ==================== 工具调用相关 ====================

class YoloDetectRequest(BaseModel):
    """YOLO检测请求"""
    image_url: str = Field(..., description="图片URL")
    conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度阈值")


class SamSegmentRequest(BaseModel):
    """SAM分割请求"""
    image_url: str = Field(..., description="图片URL")
    bbox: Optional[List[float]] = Field(default=None, description="边界框[x1,y1,x2,y2]")
    point: Optional[List[float]] = Field(default=None, description="点击点[x,y]")


class ClipEmbedRequest(BaseModel):
    """CLIP向量化请求"""
    text: Optional[str] = Field(default=None, description="文本")
    image_url: Optional[str] = Field(default=None, description="图片URL")
    mode: str = Field(default="text", description="模式: text/image/multimodal")

    @validator('text', 'image_url')
    def at_least_one_required(cls, v, values):
        if not values.get('text') and not values.get('image_url'):
            raise ValueError("text或image_url至少需要提供一个")
        return v


class DocumentParseRequest(BaseModel):
    """文档解析请求"""
    file_url: str = Field(..., description="文档URL")
    file_type: str = Field(..., description="文件类型: pdf/docx/txt")