"""
知识库检索工具

封装文本向量化 + Redis 向量检索流程，提供统一的知识检索能力。
支持按分类和标签过滤检索结果。

【调用链】
query → TextEmbedding.embed() → 1024维向量
       → VectorService.search(vector, filter=...) → TopK 相似文档（可选过滤）
       → 格式化为 List[VectorSearchResult]（含 metadata）

【关联】
- 上游：chains/retrieval_chain.py（编排检索流程）
- 下游：embeddings/text_embedding.py + services/vector_service.py
"""

from typing import List, Optional

from tools.base_tool import BaseTool, ToolException
from embeddings.text_embedding import get_text_embedding
from services.vector_service import get_vector_service
from schemas.models import VectorSearchResult


class KnowledgeRetrievalTool(BaseTool):
    """
    知识库向量检索工具

    输入查询文本 → 向量化 → Redis ANN 搜索（支持分类/标签过滤） → 返回 TopK 相似文档。
    异常通过 ToolException 向上抛出，由 BaseTool.run() 统一捕获。
    """

    @property
    def name(self) -> str:
        return "knowledge_retrieval"

    @property
    def description(self) -> str:
        return (
            "从向量知识库中检索与查询文本语义最相似的文档。"
            "支持按 category（分类）和 tags（标签）过滤。"
            "适用场景：用户询问设备知识、故障原因、维修方法等需要查资料的情况。"
        )

    @staticmethod
    def _build_filter(category: str = None, tags: List[str] = None) -> Optional[str]:
        """
        构建 RediSearch 过滤表达式

        RediSearch TAG 字段语法：
        - 单值相等: @category:{motor}
        - 多值匹配: @tags:{bearing|overheat}（OR 语义）
        - 组合过滤: (@category:{motor} @tags:{bearing})（AND 语义）
        """
        parts = []
        if category:
            parts.append(f"@category:{{{category}}}")
        if tags:
            tags_expr = "|".join(tags)
            parts.append(f"@tags:{{{tags_expr}}}")

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return f"({' '.join(parts)})"

    async def _execute(
        self,
        query: str,
        top_k: int = 5,
        category: str = None,
        tags: List[str] = None
    ) -> List[VectorSearchResult]:
        """
        执行知识检索

        Args:
            query: 查询文本（自然语言）
            top_k: 返回文档数量，默认5
            category: 分类过滤（如 "motor"），对应 Redis TAG 字段
            tags: 标签过滤（如 ["bearing", "overheat"]），OR 语义

        Returns:
            List[VectorSearchResult]: 按 score 降序排列的检索结果

        Raises:
            ToolException: EMBEDDING_FAILED / SEARCH_FAILED
        """
        # 1. 文本向量化
        try:
            embedding_service = get_text_embedding()
            vector = await embedding_service.embed(query)
        except Exception as e:
            raise ToolException(
                code="EMBEDDING_FAILED",
                message=f"文本向量化失败: {e}"
            )

        # 2. 构建过滤表达式
        filter_expr = self._build_filter(category, tags)

        # 3. Redis 向量搜索
        try:
            vector_service = get_vector_service()
            docs = vector_service.search(
                vector, top_k=top_k, include_metadata=True, filter=filter_expr
            )
        except Exception as e:
            raise ToolException(
                code="SEARCH_FAILED",
                message=f"向量检索失败: {e}"
            )

        # 4. 格式化为 VectorSearchResult 列表
        results: List[VectorSearchResult] = []
        for doc in docs:
            results.append(VectorSearchResult(
                id=doc.get("doc_id", ""),
                score=doc.get("score", 0.0),
                content=doc.get("text", ""),
                metadata=doc.get("metadata", {})
            ))

        return results


# 单例
_retrieval_tool: Optional[KnowledgeRetrievalTool] = None


def get_knowledge_retrieval_tool() -> KnowledgeRetrievalTool:
    """获取知识检索工具单例"""
    global _retrieval_tool
    if _retrieval_tool is None:
        _retrieval_tool = KnowledgeRetrievalTool()
    return _retrieval_tool
