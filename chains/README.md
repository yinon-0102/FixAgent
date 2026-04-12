# Chains 模块

## 模块概述

Chains模块封装了**LangChain Expression Language (LCEL)**链式调用，将Agent、Tools、Services串联成完整的业务流程。

与Agents模块的关系：
- **Agents**: 定义"谁来做什么"（职责和角色）
- **Chains**: 定义"怎么做"（具体执行流程、工具调用顺序）

```
Agents（定义角色）
    │
    └──► Chains（定义执行流程）
              │
              ├──► LCEL (LangChain Expression Language)
              ├──► 工具调用顺序
              ├──► 结果处理/转换
              └──► 多Agent协作
```

## 核心概念

### LangChain Expression Language (LCEL)

LCEL是一种声明式的链式调用语法，使用`|`操作符将Runnable对象串联：

```python
chain = prompt | model | output_parser
```

优点：
- **声明式**: 代码清晰，易读易维护
- **可组合**: 轻松组合出复杂流程
- **内置流式**: 原生支持流式输出
- **并行执行**: 支持`RunnableBranch`、`RunnableParallel`

## 链列表

| Chain | 文件 | 用途 | 输入 | 输出 |
|-------|------|------|------|------|
| `retrieval_chain` | retrieval_chain.py | 知识检索链 | 用户查询 | TopK相关片段 |
| `diagnosis_chain` | diagnosis_chain.py | 故障诊断链 | 故障现象 | 原因概率列表 |
| `guidance_chain` | guidance_chain.py | 作业指引链 | 诊断结果 | 标准化步骤 |
| `orchestrator_chain` | orchestrator.py | 调度链 | 用户请求 | 任务分发执行 |
| `full_pipeline_chain` | pipeline.py | 完整流程链 | 用户请求 | 完整回答 |

## 技术选型

| 组件 | 选型 | 理由 |
|-----|------|------|
| 链式调用 | LangChain Expression Language (LCEL) | 声明式、易组合 |
| 工具绑定 | `@tool` 装饰器 | 简洁、类型安全 |
| 输出解析 | Pydantic Output Parser | 结构化输出 |

## 项目中的实现

### retrieval_chain.py - 检索链

```python
# chains/retrieval_chain.py
"""
检索链

完整的知识检索流程：
1. 理解查询意图
2. 生成查询向量
3. 向量库检索
4. 结果重排序
5. 格式化输出
"""

from typing import List, Dict, Any, Optional, Callable
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from services.vector_service import get_vector_service
from services.llm_service import get_llm_service
from embeddings.multimodal_embedding import get_multimodal_embedding
from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool


# ==================== 输出模型 ====================

class RetrievalResult(BaseModel):
    """检索结果"""
    content: str = Field(description="知识内容")
    source: str = Field(description="来源")
    score: float = Field(description="相似度分数")


class RetrievalChainOutput(BaseModel):
    """检索链输出"""
    query: str = Field(description="原始查询")
    total: int = Field(description="结果总数")
    results: List[RetrievalResult] = Field(description="检索结果列表")


# ==================== 检索链实现 ====================

class RetrievalChain:
    """
    检索链

    流程：
    query → embed → vector_search → rerank → format_output
    """

    def __init__(self):
        self.vector_service = get_vector_service()
        self.llm_service = get_llm_service()
        self.multimodal = get_multimodal_embedding()

        # 构建LCEL链
        self._build_chain()

    def _build_chain(self):
        """构建LCEL链"""
        # 1. 查询理解提示词
        query_understanding_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的知识检索助手。
请理解用户的查询意图，并生成优化的检索 query。
如果用户询问过于模糊，请补充相关背景知识。"""),
            ("human", "用户查询: {query}")
        ])

        # 2. 查询理解链
        self.query_understanding_chain = (
            query_understanding_prompt
            | self.llm_service
            | StrOutputParser()
        )

        # 3. 检索提示词
        retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的设备检修知识助手。
根据检索到的内容，回答用户的问题。
如果检索内容不相关，说明无法回答并建议用户补充信息。"""),
            ("human", """用户问题: {query}

检索到的知识:
{context}

请基于以上知识回答用户问题。""")
        ])

        # 4. 检索+回答链
        self.retrieval_and_answer_chain = (
            retrieval_prompt
            | self.llm_service
            | StrOutputParser()
        )

    async def retrieve(
        self,
        query: str,
        images: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        执行检索

        Args:
            query: 用户查询
            images: 可选的图片
            top_k: 返回数量

        Returns:
            检索结果字典
        """
        # 1. 优化查询（可选）
        # understanding = await self.query_understanding_chain.ainvoke({"query": query})

        # 2. 生成查询向量
        if images:
            query_vector = await self.multimodal.embed_query(
                query_text=query,
                query_images=images
            )
        else:
            query_vector = await self.multimodal.embed_text(query)

        # 3. 向量检索
        search_results = await self.vector_service.search(
            index_name="knowledge:vectors",
            query_vector=query_vector,
            top_k=top_k
        )

        # 4. 构建上下文
        context = self._build_context(search_results)

        # 5. 生成回答
        answer = await self.retrieval_and_answer_chain.ainvoke({
            "query": query,
            "context": context
        })

        return {
            "query": query,
            "answer": answer,
            "total": len(search_results),
            "results": search_results
        }

    def _build_context(self, search_results: List[Dict]) -> str:
        """构建检索上下文"""
        if not search_results:
            return "未找到相关知识。"

        context = "## 检索到的相关知识\n\n"
        for i, result in enumerate(search_results, 1):
            context += f"### {i}. {result.get('content', '')[:200]}...\n"
            context += f"   来源: {result.get('id', '未知')}\n"
            context += f"   相似度: {result.get('score', 0):.2f}\n\n"

        return context


# 单例
_retrieval_chain: Optional[RetrievalChain] = None


def get_retrieval_chain() -> RetrievalChain:
    global _retrieval_chain
    if _retrieval_chain is None:
        _retrieval_chain = RetrievalChain()
    return _retrieval_chain
```

### diagnosis_chain.py - 诊断链

```python
# chains/diagnosis_chain.py
"""
诊断链

完整的故障诊断流程：
1. 症状分析
2. 图谱查询
3. 知识检索
4. 原因推理
5. 概率排序
"""

from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from pydantic import BaseModel, Field

from services.graph_service import get_graph_service
from services.llm_service import get_llm_service
from tools.graph_query_tool import GraphQueryTool
from tools.knowledge_retrieval_tool import KnowledgeRetrievalTool


# ==================== 输出模型 ====================

class CauseProbability(BaseModel):
    """原因概率"""
    cause: str = Field(description="故障原因")
    probability: float = Field(description="概率", ge=0.0, le=1.0)
    evidence: str = Field(description="依据")


class DiagnosisChainOutput(BaseModel):
    """诊断链输出"""
    symptom: str = Field(description="故障现象")
    possible_causes: List[CauseProbability] = Field(description="可能原因列表")
    analysis: str = Field(description="分析说明")


# ==================== 诊断链实现 ====================

class DiagnosisChain:
    """
    诊断链

    流程：
    symptom → symptom_understanding → graph_query → knowledge_query → cause_analysis → ranking
    """

    def __init__(self):
        self.graph_service = get_graph_service()
        self.llm_service = get_llm_service()
        self.graph_tool = GraphQueryTool()
        self.retrieval_tool = KnowledgeRetrievalTool()

        self._build_chain()

    def _build_chain(self):
        """构建LCEL链"""
        # 症状理解提示词
        symptom_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个设备故障诊断专家。
请分析用户描述的故障现象，提取关键症状。
使用标准化的术语描述故障类型。"""),
            ("human", "故障描述: {symptom}")
        ])

        self.symptom_understanding_chain = (
            symptom_prompt
            | self.llm_service
            | StrOutputParser()
        )

        # 原因推理提示词
        cause_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个经验丰富的设备故障诊断专家。
根据故障现象、图谱关系和历史知识，分析可能的故障原因。
按可能性从高到低排序，给出概率和依据。"""),
            ("human", """故障现象: {symptom}

图谱关联信息:
{graph_info}

历史知识:
{knowledge_info}

请分析可能的原因并排序。""")
        ])

        self.cause_analysis_chain = (
            cause_analysis_prompt
            | self.llm_service
            | StrOutputParser()
        )

    async def diagnose(
        self,
        symptom: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行诊断

        Args:
            symptom: 故障现象描述
            context: 额外上下文

        Returns:
            诊断结果
        """
        # 1. 理解症状
        understood_symptom = await self.symptom_understanding_chain.ainvoke({
            "symptom": symptom
        })

        # 2. 图谱查询
        graph_result = await self.graph_tool.execute(
            entity_name=understood_symptom,
            entity_type="Symptom",
            depth=2
        )

        # 3. 知识检索
        knowledge_result = await self.retrieval_tool.execute(
            query=symptom,
            top_k=3
        )

        # 4. 原因分析
        graph_info = self._format_graph_result(graph_result)
        knowledge_info = self._format_knowledge_result(knowledge_result)

        cause_analysis = await self.cause_analysis_chain.ainvoke({
            "symptom": understood_symptom,
            "graph_info": graph_info,
            "knowledge_info": knowledge_info
        })

        return {
            "original_symptom": symptom,
            "understood_symptom": understood_symptom,
            "analysis": cause_analysis,
            "graph_relations": graph_result.data if graph_result.status == "success" else [],
            "knowledge_refs": knowledge_result.data if knowledge_result.status == "success" else []
        }

    def _format_graph_result(self, result) -> str:
        """格式化图谱结果"""
        if result.status != "success":
            return "图谱查询失败"

        nodes = result.data.get("nodes", [])
        relations = result.data.get("relations", [])

        info = f"找到 {len(nodes)} 个相关实体，{len(relations)} 条关系\n\n"

        for node in nodes[:5]:
            info += f"- {node.get('label')}: {node.get('properties', {}).get('name', '')}\n"

        return info

    def _format_knowledge_result(self, result) -> str:
        """格式化知识结果"""
        if result.status != "success":
            return "知识查询失败"

        results = result.data.get("results", [])
        info = f"找到 {len(results)} 条相关知识\n\n"

        for item in results[:3]:
            info += f"- {item.get('content', '')[:100]}...\n"

        return info


# 单例
_diagnosis_chain: Optional[DiagnosisChain] = None


def get_diagnosis_chain() -> DiagnosisChain:
    global _diagnosis_chain
    if _diagnosis_chain is None:
        _diagnosis_chain = DiagnosisChain()
    return _diagnosis_chain
```

### guidance_chain.py - 作业指引链

```python
# chains/guidance_chain.py
"""
作业指引链

完整的维修作业指引流程：
1. 获取诊断结果
2. 查询标准模板
3. 填充个性化内容
4. 安全合规检查
5. 生成步骤指引
"""

from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from pydantic import BaseModel, Field

from services.llm_service import get_llm_service


# ==================== 输出模型 ====================

class RepairStep(BaseModel):
    """维修步骤"""
    step_number: int = Field(description="步骤序号")
    action: str = Field(description="操作动作")
    checkpoints: List[str] = Field(description="检查点")
    safety_notes: List[str] = Field(description="安全注意事项")
    tools_needed: List[str] = Field(description="所需工具")


class GuidanceChainOutput(BaseModel):
    """作业指引链输出"""
    diagnosis: str = Field(description="诊断结果")
    steps: List[RepairStep] = Field(description="维修步骤")
    estimated_time: str = Field(description="预计时间")
    warnings: List[str] = Field(description="重要警告")


# ==================== 作业指引链实现 ====================

class GuidanceChain:
    """
    作业指引链

    流程：
    diagnosis → knowledge_retrieval → content_fill → safety_check → format_steps
    """

    def __init__(self):
        self.llm_service = get_llm_service()

        self._build_chain()

    def _build_chain(self):
        """构建LCEL链"""
        # 步骤生成提示词
        step_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个标准作业流程专家。
根据诊断结果，生成标准化的维修步骤。

要求：
1. 步骤按正确顺序排列
2. 每步包含：动作、检查点、安全注意、所需工具
3. 符合行业安全规范
4. 适合现场执行

输出格式：
### 维修步骤

**步骤1: [动作描述]**
- 检查点: [检查内容]
- 安全注意: [安全事项]
- 所需工具: [工具列表]

...

**预计时间**: [时间]
**重要警告**: [如有]
"""),
            ("human", """诊断结果: {diagnosis}

相关知识:
{knowledge}

请生成维修步骤。""")
        ])

        self.step_generation_chain = (
            step_generation_prompt
            | self.llm_service
            | StrOutputParser()
        )

    async def generate_guidance(
        self,
        diagnosis: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成作业指引

        Args:
            diagnosis: 诊断结果
            context: 额外上下文

        Returns:
            作业指引
        """
        # 1. 生成步骤
        knowledge_content = context.get("knowledge", "无相关信息") if context else "无相关信息"

        steps = await self.step_generation_chain.ainvoke({
            "diagnosis": diagnosis,
            "knowledge": knowledge_content
        })

        return {
            "diagnosis": diagnosis,
            "steps": steps
        }


# 单例
_guidance_chain: Optional[GuidanceChain] = None


def get_guidance_chain() -> GuidanceChain:
    global _guidance_chain
    if _guidance_chain is None:
        _guidance_chain = GuidanceChain()
    return _guidance_chain
```

### orchestrator.py - 调度链

```python
# chains/orchestrator.py
"""
调度链

基于LCEL实现的多Agent调度链。
支持：
- 条件分支（根据意图选择不同路径）
- 并行执行（多个Agent同时工作）
- 结果聚合（汇总各Agent结果）
"""

from typing import Dict, Any, Optional, List, Union
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import (
    RunnableBranch,
    RunnableParallel,
    RunnablePassthrough
)
from pydantic import BaseModel, Field

from services.llm_service import get_llm_service
from .retrieval_chain import get_retrieval_chain
from .diagnosis_chain import get_diagnosis_chain
from .guidance_chain import get_guidance_chain


class OrchestratorChain:
    """
    调度链

    使用LCEL的RunnableBranch实现意图分派：
    - 检索意图 → retrieval_chain
    - 诊断意图 → diagnosis_chain
    - 作业意图 → guidance_chain
    - 混合意图 → full_pipeline
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        self.retrieval_chain = get_retrieval_chain()
        self.diagnosis_chain = get_diagnosis_chain()
        self.guidance_chain = get_guidance_chain()

        self._build_chain()

    def _build_chain(self):
        """构建LCEL链"""
        # 意图识别提示词
        intention_prompt = ChatPromptTemplate.from_messages([
            ("system", """分析用户意图，返回最合适的模式。

模式选项：
- retrieval: 用户想查询知识、了解信息
- diagnosis: 用户遇到故障，需要诊断原因
- guidance: 用户需要维修步骤、操作指引
- full: 用户需要完整帮助（诊断+指引）
- chat: 一般对话

只返回一个词：retrieval/diagnosis/guidance/full/chat"""),
            ("human", "用户消息: {message}")
        ])

        # 意图识别链
        self.intention_chain = (
            intention_prompt
            | self.llm_service
            | StrOutputParser()
        )

        # 定义各分支链
        retrieval_branch = (
            RunnableBranch(
                # 条件: intention == "retrieval"
                lambda x: "retrieval" in x.get("intention", "").lower(),
                self._create_retrieval_subchain(),
                # 默认
                RunnablePassthrough()
            )
        )

        # 构建主链
        self.main_chain = (
            RunnableParallel(
                # 并行识别意图和处理
                intention=lambda x: self.intention_chain.ainvoke({"message": x["message"]}),
                original_input=RunnablePassthrough()
            )
            | RunnablePassthrough.assign(
                result=self._route_and_execute()
            )
        )

    def _create_retrieval_subchain(self):
        """创建检索子链"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的设备检修知识助手。"),
            ("human", "{query}")
        ])

        return (
            prompt
            | self.llm_service
            | StrOutputParser()
        )

    def _route_and_execute(self):
        """根据意图路由执行"""
        async def route(x: Dict) -> Dict:
            intention = x.get("intention", "").lower()
            original = x.get("original_input", {})
            query = original.get("message", "")

            result = {"message": query}

            if "retrieval" in intention or "knowledge" in intention:
                # 执行检索
                result = await self.retrieval_chain.retrieve(query)
            elif "diagnosis" in intention or "故障" in intention:
                # 执行诊断
                result = await self.diagnosis_chain.diagnose(query)
            elif "guidance" in intention or "指引" in intention:
                # 执行指引
                result = await self.guidance_chain.generate_guidance(query)
            elif "full" in intention:
                # 完整流程
                result = await self._run_full_pipeline(query, original)
            else:
                # 简单对话
                chat_result = await self.llm_service.chat([
                    {"role": "user", "content": query}
                ])
                result = {"message": chat_result["content"]}

            return result

        return route

    async def _run_full_pipeline(self, query: str, context: Dict) -> Dict:
        """执行完整流程"""
        # 1. 诊断
        diagnosis_result = await self.diagnosis_chain.diagnose(query)

        # 2. 指引
        guidance_result = await self.guidance_chain.generate_guidance(
            diagnosis=diagnosis_result.get("analysis", query),
            context={"symptom": diagnosis_result.get("understood_symptom", "")}
        )

        # 3. 汇总
        return {
            "diagnosis": diagnosis_result.get("analysis", ""),
            "guidance": guidance_result.get("steps", ""),
            "symptom": diagnosis_result.get("understood_symptom", "")
        }

    async def run(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行调度链

        Args:
            message: 用户消息
            context: 额外上下文

        Returns:
            执行结果
        """
        return await self.main_chain.ainvoke({
            "message": message,
            "context": context or {}
        })


# 单例
_orchestrator_chain: Optional[OrchestratorChain] = None


def get_orchestrator_chain() -> OrchestratorChain:
    global _orchestrator_chain
    if _orchestrator_chain is None:
        _orchestrator_chain = OrchestratorChain()
    return _orchestrator_chain
```

### pipeline.py - 完整流程链

```python
# chains/pipeline.py
"""
完整流程管道

将检索、诊断、指引串联成完整的业务流程。
用于'完整模式'（FULL），一次调用完成所有分析。
"""

from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from services.llm_service import get_llm_service
from .retrieval_chain import get_retrieval_chain
from .diagnosis_chain import get_diagnosis_chain
from .guidance_chain import get_guidance_chain


class FullPipelineChain:
    """
    完整流程管道

    串联三个子链：
    retrieval → diagnosis → guidance → summary
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        self.retrieval_chain = get_retrieval_chain()
        self.diagnosis_chain = get_diagnosis_chain()
        self.guidance_chain = get_guidance_chain()

        self._build_summary_chain()

    def _build_summary_chain(self):
        """构建汇总链"""
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的设备检修助手。
请将检索、诊断、指引的结果汇总成一份完整的报告。

格式：
## 检修报告

### 问题确认
{diagnosis}

### 维修步骤
{guidance}

### 相关知识
{retrieval}

---
*如有疑问，请联系专业技术人员。*
"""),
            ("human", """请汇总以下信息：""")
        ])

        self.summary_chain = (
            summary_prompt
            | self.llm_service
            | StrOutputParser()
        )

    async def run(
        self,
        query: str,
        images: Optional[list] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行完整流程

        Args:
            query: 用户问题
            images: 可选的图片
            context: 额外上下文

        Returns:
            完整报告
        """
        results = {}

        # 1. 检索相关知识
        retrieval_result = await self.retrieval_chain.retrieve(
            query=query,
            images=images,
            top_k=5
        )
        results["retrieval"] = retrieval_result.get("answer", "")

        # 2. 诊断故障原因
        diagnosis_result = await self.diagnosis_chain.diagnose(
            symptom=query,
            context=context
        )
        results["diagnosis"] = diagnosis_result.get("analysis", "")

        # 3. 生成维修指引
        guidance_result = await self.guidance_chain.generate_guidance(
            diagnosis=results["diagnosis"],
            context={
                "symptom": diagnosis_result.get("understood_symptom", ""),
                "knowledge": results["retrieval"]
            }
        )
        results["guidance"] = guidance_result.get("steps", "")

        # 4. 汇总生成最终报告
        final_report = await self.summary_chain.ainvoke({
            "diagnosis": results["diagnosis"],
            "guidance": results["guidance"],
            "retrieval": results["retrieval"]
        })

        results["final_report"] = final_report

        return results


# 单例
_full_pipeline_chain: Optional[FullPipelineChain] = None


def get_full_pipeline_chain() -> FullPipelineChain:
    global _full_pipeline_chain
    if _full_pipeline_chain is None:
        _full_pipeline_chain = FullPipelineChain()
    return _full_pipeline_chain
```

## 使用示例

### 1. 单独使用检索链

```python
from chains import get_retrieval_chain

retrieval_chain = get_retrieval_chain()

result = await retrieval_chain.retrieve(
    query="电动机轴承过热",
    top_k=10
)

print(result["answer"])
```

### 2. 使用完整流程链

```python
from chains import get_full_pipeline_chain

pipeline = get_full_pipeline_chain()

result = await pipeline.run(
    query="我的电动机轴承最近老是过热，是怎么回事？",
    images=["http://example.com/fault.jpg"]
)

print(result["final_report"])
```

### 3. LCEL组合自定义链

```python
from langchain.schema.runnable import RunnableParallel, RunnableBranch

# 创建自定义组合链
custom_chain = (
    RunnableParallel(
        retrieval=retrieval_chain.retrieve,
        diagnosis=diagnosis_chain.diagnose
    )
    | RunnableBranch(
        # 如果诊断发现问题，生成指引
        lambda x: "故障" in x.get("diagnosis", ""),
        guidance_chain.generate_guidance,
        # 否则只返回诊断
        RunnablePassthrough()
    )
)
```

## LCEL核心语法

| 语法 | 说明 | 示例 |
|-----|------|------|
| `A \| B` | 管道操作符，A输出传给B | `prompt \| model \| parser` |
| `RunnableParallel` | 并行执行多个Runnable | `RunnableParallel(a=chain1, b=chain2)` |
| `RunnableBranch` | 条件分支 | `RunnableBranch((cond, action), ...)` |
| `RunnablePassthrough` | 透传输入 | `RunnablePassthrough.assign(...)` |
| `.invoke()` | 同步调用 | `chain.invoke({"query": "..."})` |
| `.ainvoke()` | 异步调用 | `await chain.ainvoke({"query": "..."})` |
| `.stream()` | 同步流式 | `for chunk in chain.stream(...):` |
| `.astream()` | 异步流式 | `async for chunk in chain.astream(...):` |

## 文件结构

```
chains/
├── __init__.py
├── README.md                    # 本文件
├── retrieval_chain.py           # 检索链
├── diagnosis_chain.py           # 诊断链
├── guidance_chain.py            # 作业指引链
├── orchestrator.py              # 调度链
└── pipeline.py                 # 完整流程链
```

## 注意事项

1. **链的复用**: 使用单例模式避免重复创建链对象
2. **错误处理**: 每个链内部应捕获异常，返回友好的错误信息
3. **流式支持**: LCEL原生支持流式，但多链协作时需特殊处理
4. **上下文传递**: 使用`RunnablePassthrough.assign()`传递额外上下文
5. **性能优化**: 独立步骤可并行执行，使用`RunnableParallel`
