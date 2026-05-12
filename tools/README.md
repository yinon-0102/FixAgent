# Tools 模块

## 模块概述

Tools 模块是Agent的**能力扩展层**，封装了系统可调用的所有工具。本模块将各种外部能力（检测、分割、检索等）封装为统一接口，供Agent在执行过程中调用。

核心价值：
- **能力抽象**: 将复杂能力简化为Tool调用
- **可组合**: 多个Tool可组合使用完成复杂任务
- **可扩展**: 新增Tool只需实现基类接口

## 工具列表

| 工具 | 描述 | 输入 | 输出 | 优先级 |
|-----|------|------|------|-------|
| `knowledge_retrieval` | 知识库向量检索 | 查询文本 + 分类/标签过滤 | TopK相关片段 + metadata | P0 |
| `fact_retrieval` | 事实向量检索 | 关键词列表 | 相似历史事实 | P0 |
| `graph_query` | 图谱关系查询 | 实体/关系 | 关联实体列表 | P0 |
| `yolo_detect` | YOLO目标检测 | 图片 | 检测结果列表 | P0 |
| `sam_segment` | SAM图像分割 | 图片+点/框 | 分割掩码+类别 | P1 |
| `document_parser` | 文档解析 | PDF/图片路径 | 文本+表格+图片 | P1 |

## 技术选型

| 组件 | 选型 | 理由 |
|-----|------|------|
| 目标检测 | YOLOv8 | 速度快、预训练模型、社区活跃 |
| 图像分割 | SAM (Segment Anything) | Meta开源、高精度分割 |
| 文档解析 | PyMuPDF + PaddleOCR | 中文支持好 |
| 工具框架 | LangChain @tool | 集成方便、类型安全 |

## 项目中的实现

### base_tool.py - 工具基类

采用**模板方法模式**，`run()` 统一处理异常 → ToolResult，子类只需实现 `_execute()` 写正常业务逻辑。
预期失败抛 `ToolException(code, message)`，未知异常自动捕获为 `code="TOOL_ERROR"`。

```python
# tools/base_tool.py
"""
工具基类模块 — 统一工具接口与返回模型
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel, Field


class ToolError(BaseModel):
    """工具执行错误模型"""
    code: str = Field(description="错误码，机器可读，如 EMBEDDING_FAILED")
    message: str = Field(description="错误描述，人类可读")


class ToolResult(BaseModel):
    """工具执行结果模型"""
    success: bool = Field(description="是否执行成功")
    data: Any = Field(default=None, description="成功时的返回数据")
    error: Optional[ToolError] = Field(default=None, description="失败时的错误信息")
    tool_name: str = Field(description="来源工具名")


class ToolException(Exception):
    """工具执行异常 — 子类在 _execute() 中遇到业务错误时抛出，携带错误码"""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class BaseTool(ABC):
    """工具抽象基类 — 模板方法 run() 统一异常兜底"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """核心业务逻辑，只写正常流程。失败抛 ToolException(code, message)"""
        pass

    async def run(self, **kwargs) -> ToolResult:
        """模板方法：自动 try/execute/catch，子类无需手动包装"""
        try:
            data = await self._execute(**kwargs)
            return ToolResult(success=True, data=data, tool_name=self.name)
        except ToolException as e:
            return ToolResult(
                success=False, tool_name=self.name,
                error=ToolError(code=e.code, message=e.message)
            )
        except Exception as e:
            return ToolResult(
                success=False, tool_name=self.name,
                error=ToolError(code="TOOL_ERROR", message=str(e))
            )

```

### knowledge_retrieval_tool.py - 知识检索工具

```python
# tools/knowledge_retrieval_tool.py
"""
知识检索工具

基于向量检索从知识库中查找相关内容。
支持：
- 纯文本检索
- 纯图片检索
- 图文混合检索
- 元数据过滤
"""

from typing import List, Optional, Dict, Any
from pydantic import Field
import httpx

from .base_tool import BaseTool
from services.vector_service import get_vector_service
from services.llm_service import get_llm_service
from embeddings.multimodal_embedding import get_multimodal_embedding
from config.settings import get_settings


class KnowledgeRetrievalTool(BaseTool):
    """
    知识检索工具

    从向量知识库中检索与查询相关的内容。
    """

    def __init__(self):
        self.vector_service = get_vector_service()
        self.llm_service = get_llm_service()
        self.multimodal = get_multimodal_embedding()
        self.settings = get_settings()

    @property
    def name(self) -> str:
        return "knowledge_retrieval"

    @property
    def description(self) -> str:
        return """
        知识检索工具 - 从设备检修知识库中检索相关信息。

        用途：
        - 根据用户描述的问题检索相关知识
        - 查找设备故障的解决方案
        - 获取检修标准和规范

        输入：
        - query: 检索查询文本（如"电动机轴承过热"）
        - images: 可选的图片URL列表，用于图片辅助检索
        - top_k: 返回结果数量，默认10

        输出：
        - results: 检索到的知识片段列表
        - 每个结果包含：content（内容）、score（相似度）、source（来源）

        使用场景：
        - 用户询问故障原因时
        - 需要查找检修方案时
        - 不确定具体问题，需要泛化检索时
        """

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询文本"
                },
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选的图片URL列表"
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "返回结果数量"
                },
                "category": {
                    "type": "string",
                    "description": "可选的分类过滤"
                }
            },
            "required": ["query"]
        }

    async def _execute(
        self,
        query: str,
        images: Optional[List[str]] = None,
        top_k: int = 10,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行知识检索

        流程：
        1. 将查询文本/图片转换为向量
        2. 在Redis向量库中检索
        3. 返回TopK结果
        """
        # 1. 生成查询向量
        if images:
            # 图文混合查询
            query_vector = await self.multimodal.embed_query(
                query_text=query,
                query_images=images
            )
        else:
            # 纯文本查询
            query_vector = await self.multimodal.embed_text(query)

        # 2. 向量检索
        index_name = self.settings.get("VECTOR_INDEX", "knowledge:vectors")
        results = await self.vector_service.search(
            index_name=index_name,
            query_vector=query_vector,
            top_k=top_k
        )

        # 3. 格式化结果
        formatted_results = []
        for item in results:
            formatted_results.append({
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "id": item.get("id", ""),
                "metadata": item.get("metadata", {})
            })

        return {
            "query": query,
            "total": len(formatted_results),
            "results": formatted_results
        }
```

### graph_query_tool.py - 图谱查询工具

```python
# tools/graph_query_tool.py
"""
图谱查询工具

基于Neo4j图数据库进行关系查询。
支持：
- 实体查询
- 关系扩展
- 路径查找
"""

from typing import List, Optional, Dict, Any
from pydantic import Field

from .base_tool import BaseTool
from services.graph_service import get_graph_service


class GraphQueryTool(BaseTool):
    """
    图谱查询工具

    从Neo4j图数据库中查询实体和关系。
    """

    def __init__(self):
        self.graph_service = get_graph_service()

    @property
    def name(self) -> str:
        return "graph_query"

    @property
    def description(self) -> str:
        return """
        图谱查询工具 - 查询设备故障知识图谱中的实体和关系。

        用途：
        - 查询设备-部件-故障现象的关联关系
        - 查找故障原因到解决方案的路径
        - 扩展查询相关实体

        输入：
        - entity_name: 实体名称（如"轴承过热"）
        - entity_type: 实体类型（Device/Component/Symptom/Cause/Solution/Step）
        - relation_type: 关系类型（CONTAINS/MANIFESTS_AS/CAUSED_BY/RESOLVED_BY/INCLUDES）
        - depth: 查询深度，默认1

        输出：
        - nodes: 查询到的节点列表
        - relations: 查询到的关系列表

        使用场景：
        - 需要理解故障的因果链条时
        - 诊断时需要查找关联原因时
        - 需要追溯完整解决方案时
        """

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "实体名称"
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["Device", "Component", "Symptom", "Cause", "Solution", "Step"],
                    "description": "实体类型"
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["CONTAINS", "MANIFESTS_AS", "CAUSED_BY", "RESOLVED_BY", "INCLUDES"],
                    "description": "关系类型"
                },
                "depth": {
                    "type": "integer",
                    "default": 1,
                    "description": "查询深度"
                }
            },
            "required": ["entity_name"]
        }

    async def _execute(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        depth: int = 1
    ) -> Dict[str, Any]:
        """执行图谱查询"""
        # 扩展查询相关实体
        result = await self.graph_service.expand_from_entity(
            entity_name=entity_name,
            entity_label=entity_type or "Entity",
            max_depth=depth
        )

        return {
            "entity_name": entity_name,
            "nodes": result.get("nodes", []),
            "relations": result.get("relations", []),
            "total_nodes": len(result.get("nodes", [])),
            "total_relations": len(result.get("relations", []))
        }


class GraphPathTool(BaseTool):
    """
    图谱路径查询工具

    查找两个实体之间的最短路径。
    """

    def __init__(self):
        self.graph_service = get_graph_service()

    @property
    def name(self) -> str:
        return "graph_path"

    @property
    def description(self) -> str:
        return """
        图谱路径查询工具 - 查找两个实体之间的关联路径。

        用途：
        - 查找故障原因到解决方案的完整路径
        - 追溯设备到故障现象的链条

        输入：
        - source_name: 起点实体名称
        - target_name: 终点实体名称
        - source_type: 起点类型（可选）
        - target_type: 终点类型（可选）

        输出：
        - path: 从起点到终点的节点序列
        - length: 路径长度（跳数）

        使用场景：
        - 已知故障原因，需要找到解决方案时
        - 需要展示完整的因果链条时
        """

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_name": {
                    "type": "string",
                    "description": "起点实体名称"
                },
                "target_name": {
                    "type": "string",
                    "description": "终点实体名称"
                },
                "source_type": {
                    "type": "string",
                    "description": "起点实体类型"
                },
                "target_type": {
                    "type": "string",
                    "description": "终点实体类型"
                }
            },
            "required": ["source_name", "target_name"]
        }

    async def _execute(
        self,
        source_name: str,
        target_name: str,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行路径查询"""
        path = await self.graph_service.find_shortest_path(
            source_name=source_name,
            target_name=target_name,
            source_label=source_type or "Entity",
            target_label=target_type or "Entity"
        )

        if path:
            return {
                "found": True,
                "path": path,
                "length": len(path)
            }
        else:
            return {
                "found": False,
                "path": [],
                "length": 0
            }
```

### yolo_tool.py - YOLO检测工具

```python
# tools/yolo_tool.py
"""
YOLO目标检测工具

使用YOLOv8进行目标检测，识别图片中的部件和故障区域。
"""

from typing import List, Optional, Dict, Any
from pydantic import Field
import httpx

from .base_tool import BaseTool, RemoteTool
from config.settings import get_settings


class YoloDetectTool(RemoteTool):
    """
    YOLO检测工具（远程调用）

    通过调用Python微服务执行YOLO检测。
    """

    def __init__(self):
        settings = get_settings()
        super().__init__(
            service_url=settings.python_tools_url,
            timeout=30
        )

    @property
    def name(self) -> str:
        return "yolo_detect"

    @property
    def description(self) -> str:
        return """
        YOLO目标检测工具 - 识别图片中的部件和故障区域。

        用途：
        - 识别故障图片中的设备部件
        - 检测异常区域位置
        - 为后续分析提供部件信息

        输入：
        - image_url: 图片URL或本地路径
        - conf_threshold: 置信度阈值，默认0.5

        输出：
        - detections: 检测结果列表
          - class_name: 类别名称
          - confidence: 置信度
          - bbox: 边界框坐标[x1, y1, x2, y2]

        使用场景：
        - 用户上传故障图片时
        - 需要识别图片中的具体部件时
        - 作为SAM分割的前置步骤

        注意：
        - 高置信度(>=0.8)结果可直接使用
        - 低置信度结果建议配合SAM精细分割
        """

    @property
    def remote_endpoint(self) -> str:
        return "/tools/yolo"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "图片URL或本地路径"
                },
                "conf_threshold": {
                    "type": "number",
                    "default": 0.5,
                    "description": "置信度阈值"
                }
            },
            "required": ["image_url"]
        }

    async def _execute(
        self,
        image_url: str,
        conf_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """执行YOLO检测"""
        response = await httpx.AsyncClient().post(
            f"{self.service_url}{self.remote_endpoint}",
            json={
                "image_url": image_url,
                "conf_threshold": conf_threshold
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
```

### sam_tool.py - SAM分割工具

```python
# tools/sam_tool.py
"""
SAM图像分割工具

使用Segment Anything Model进行精细图像分割。
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import Field
import httpx

from .base_tool import BaseTool, RemoteTool
from config.settings import get_settings


class SamSegmentTool(RemoteTool):
    """
    SAM分割工具（远程调用）

    通过调用Python微服务执行SAM分割。
    """

    def __init__(self):
        settings = get_settings()
        super().__init__(
            service_url=settings.python_tools_url,
            timeout=60
        )

    @property
    def name(self) -> str:
        return "sam_segment"

    @property
    def description(self) -> str:
        return """
        SAM图像分割工具 - 对图像进行精细分割。

        用途：
        - 精确分割目标区域
        - 提取部件轮廓
        - 生成分割掩码

        输入：
        - image_url: 图片URL或本地路径
        - bbox: 边界框 [x1, y1, x2, y2]（可选）
        - point: 点击点 [x, y]（可选）
        - label: 点击点标签，1表示前景，0表示背景

        输出：
        - masks: 分割掩码列表
        - scores: 各掩码的置信度
        - labels: 各掩码对应的类别

        使用场景：
        - YOLO检测置信度低时
        - 需要精确分割部件时
        - 提取故障区域用于CLIP识别

        注意：
        - bbox和point至少提供一个
        - 分割结果可用于CLIP向量化
        """

    @property
    def remote_endpoint(self) -> str:
        return "/tools/sam"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "图片URL或本地路径"
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "边界框 [x1, y1, x2, y2]"
                },
                "point": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "点击点 [x, y]"
                },
                "point_label": {
                    "type": "integer",
                    "default": 1,
                    "description": "点标签，1=前景，0=背景"
                }
            },
            "required": ["image_url"]
        }

    async def _execute(
        self,
        image_url: str,
        bbox: Optional[List[float]] = None,
        point: Optional[List[float]] = None,
        point_label: int = 1
    ) -> Dict[str, Any]:
        """执行SAM分割"""
        payload = {"image_url": image_url}

        if bbox:
            payload["bbox"] = bbox
        elif point:
            payload["point"] = point
            payload["point_label"] = point_label

        response = await httpx.AsyncClient().post(
            f"{self.service_url}{self.remote_endpoint}",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
```

### document_parser_tool.py - 文档解析工具

```python
# tools/document_parser_tool.py
"""
文档解析工具

解析PDF和图片文档，提取文本、表格和图片。
"""

from typing import List, Optional, Dict, Any
from pydantic import Field
import httpx

from .base_tool import BaseTool, RemoteTool
from config.settings import get_settings


class DocumentParserTool(RemoteTool):
    """
    文档解析工具（远程调用）

    通过调用Python微服务解析文档。
    """

    def __init__(self):
        settings = get_settings()
        super().__init__(
            service_url=settings.python_tools_url,
            timeout=120  # 文档解析可能较慢
        )

    @property
    def name(self) -> str:
        return "document_parser"

    @property
    def description(self) -> str:
        return """
        文档解析工具 - 解析PDF和图片文档。

        用途：
        - 提取设备检修手册内容
        - 解析技术规格文档
        - 提取表格数据

        输入：
        - file_url: 文档URL或本地路径
        - file_type: 文件类型（pdf/docx/txt）

        输出：
        - pages: 页面内容列表
          - text: 页面文本
          - tables: 表格列表
          - images: 图片列表
        - total_pages: 总页数

        使用场景：
        - 上传检修手册时
        - 批量导入知识时
        - 需要从PDF中提取内容时

        注意：
        - PDF文件可能较大，解析需要一定时间
        - 建议先上传到OSS再解析
        """

    @property
    def remote_endpoint(self) -> str:
        return "/tools/parse"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_url": {
                    "type": "string",
                    "description": "文档URL或本地路径"
                },
                "file_type": {
                    "type": "string",
                    "enum": ["pdf", "docx", "txt"],
                    "description": "文件类型"
                }
            },
            "required": ["file_url", "file_type"]
        }

    async def _execute(
        self,
        file_url: str,
        file_type: str
    ) -> Dict[str, Any]:
        """执行文档解析"""
        response = await httpx.AsyncClient().post(
            f"{self.service_url}{self.remote_endpoint}",
            json={
                "file_url": file_url,
                "file_type": file_type
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
```

## 工具注册与使用

### LangChain Tool注册

```python
# tools/__init__.py
from langchain.tools import tool
from .knowledge_retrieval_tool import KnowledgeRetrievalTool
from .graph_query_tool import GraphQueryTool, GraphPathTool
from .yolo_tool import YoloDetectTool
from .sam_tool import SamSegmentTool
from .document_parser_tool import DocumentParserTool

# 注册工具
def get_all_tools():
    """获取所有已注册的工具"""
    return [
        KnowledgeRetrievalTool(),
        GraphQueryTool(),
        GraphPathTool(),
        YoloDetectTool(),
        SamSegmentTool(),
        DocumentParserTool(),
    ]

# LangChain格式导出
def get_langchain_tools():
    """获取LangChain格式的工具列表"""
    tools = get_all_tools()
    return [tool_wrapper(t) for t in tools]

def tool_wrapper(base_tool: BaseTool):
    """将BaseTool包装为LangChain Tool"""

    @tool
    async def langchain_tool(**kwargs):
        result = await base_tool.execute(**kwargs)
        return result.data

    langchain_tool.name = base_tool.name
    langchain_tool.description = base_tool.description

    return langchain_tool
```

### Agent中使用工具

```python
# agents/diagnosis_agent.py
from langchain.agents import initialize_agent, AgentType
from tools import get_langchain_tools

tools = get_langchain_tools()

# 创建Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 执行诊断
result = await agent.arun(
    "电动机轴承过热，可能是什么原因？",
    callbacks=[StreamingCallbackHandler()]
)
```

## 文件结构

```
tools/
├── __init__.py                     # 工具注册
├── README.md                       # 本文件
├── base_tool.py                    # 工具基类定义
├── knowledge_retrieval_tool.py     # 知识库检索工具
├── fact_retrieval_tool.py          # 事实检索工具（LLM function calling 用）
├── graph_query_tool.py             # 图谱查询工具
├── yolo_tool.py                   # YOLO 目标检测
├── sam_tool.py                    # SAM 图像分割
└── document_tool.py               # 文档解析工具
```

## 注意事项

1. **远程调用**: YOLO/SAM等重计算任务通过HTTP调用Python微服务
2. **超时处理**: 所有远程调用设置合理超时，避免Agent卡死
3. **错误恢复**: 工具执行失败时返回错误信息，不阻塞Agent
4. **工具选择**: Agent应根据意图自动选择合适的工具
