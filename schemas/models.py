"""
Schemas基础模型模块
定义枚举类型、通用常量、以及跨模块复用的基础类型。
"""

from enum import Enum
from typing import List, Any, Dict
from pydantic import BaseModel, Field


# ==================== 枚举类型 ====================

class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"           # 管理员
    USER = "user"             # 普通用户
    AUDITOR = "auditor"       # 审核员


class KnowledgeStatus(str, Enum):
    """知识条目状态"""
    DRAFT = "draft"           # 草稿
    PUBLISHED = "published"   # 已发布
    ARCHIVED = "archived"     # 已归档


class CaseStatus(str, Enum):
    """案例状态"""
    SUBMITTED = "submitted"    # 已提交
    REVIEWING = "reviewing"   # 审核中
    APPROVED = "approved"     # 已通过
    REJECTED = "rejected"     # 已拒绝


class AgentMode(str, Enum):
    """Agent运行模式"""
    CHAT = "chat"            # 对话模式
    RETRIEVAL = "retrieval"  # 检索模式
    DIAGNOSIS = "diagnosis"   # 诊断模式
    GUIDANCE = "guidance"    # 作业指引模式
    FULL = "full"             # 完整流程


class IntentionType(str, Enum):
    """用户意图类型"""
    QUERY_KNOWLEDGE = "query_knowledge"      # 查询知识
    TROUBLESHOOT = "troubleshoot"            # 故障排查
    SEEK_GUIDANCE = "seek_guidance"         # 寻求指导
    SUBMIT_CASE = "submit_case"              # 提交案例
    GENERAL_CHAT = "general_chat"           # 一般对话


class ImageProcessStatus(str, Enum):
    """图片处理状态"""
    PENDING = "pending"      # 待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败


class TaskStatus(str, Enum):
    """异步任务状态"""
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"

# ==================== 基础响应模型 ====================

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = True
    message: str = "操作成功"
    code: int = 200

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "操作成功",
                "code": 200
            }
        }


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = False
    message: str = "操作失败"
    code: int = 500

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "参数错误",
                "code": 400
            }
        }


class PaginationMeta(BaseModel):
    """分页元信息"""
    page: int = Field(default=1, ge=1, description="当前页码")
    page_size: int = Field(default=10, ge=1, le=100, description="每页数量")
    total: int = Field(default=0, description="总数")
    total_pages: int = Field(default=0, description="总页数")

    @classmethod
    def create(cls, total: int, page: int, page_size: int) -> "PaginationMeta":
        """创建分页元信息"""
        total_pages = (total + page_size - 1) // page_size
        return cls(total=total, page=page, page_size=page_size, total_pages=total_pages)


# ==================== 通用数据结构 ====================

class DetectionBox(BaseModel):
    """检测框"""
    x1: float = Field(description="左上角X坐标")
    y1: float = Field(description="左上角Y坐标")
    x2: float = Field(description="右下角X坐标")
    y2: float = Field(description="右下角Y坐标")

    def to_xyxy(self) -> List[float]:
        """转换为[x1, y1, x2, y2]格式"""
        return [self.x1, self.y1, self.x2, self.y2]


class DetectionResult(BaseModel):
    """检测结果"""
    class_name: str = Field(description="类别名称")
    confidence: float = Field(description="置信度", ge=0.0, le=1.0)
    bbox: DetectionBox = Field(description="检测框")


class VectorSearchResult(BaseModel):
    """向量搜索结果"""
    id: str = Field(description="向量ID")
    score: float = Field(description="相似度分数")
    content: str = Field(description="关联内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class GraphNode(BaseModel):
    """图谱节点"""
    id: str = Field(description="节点ID")
    label: str = Field(description="节点标签")
    properties: Dict[str, Any] = Field(default_factory=dict, description="节点属性")


class GraphRelation(BaseModel):
    """图谱关系"""
    source_id: str = Field(description="源节点ID")
    target_id: str = Field(description="目标节点ID")
    relation_type: str = Field(description="关系类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="关系属性")


class GraphQueryResult(BaseModel):
    """图谱查询结果"""
    nodes: List[GraphNode] = Field(default_factory=list)
    relations: List[GraphRelation] = Field(default_factory=list)
