"""
文本向量化模块

使用阿里云百炼 text-embedding-v4 将文本转为高维向量
"""

import hashlib
import logging
import redis
import httpx
from typing import Optional, List, Dict, Any
from config.settings import get_settings

logger = logging.getLogger(__name__)


class TextEmbedding:
    """
    文本向量化服务

    封装阿里云百炼 text-embedding-v4 API
    带 Redis 缓存
    """

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.dashscope_api_key
        self.model = "text-embedding-v4"
        self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        # Redis 缓存
        self.redis = redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.redis_password,
            db=self.settings.redis_db,
            decode_responses=False
        )
        self.cache_ttl = self.settings.redis_ttl

    def _get_cache_key(self, text: str) -> str:
        """生成缓存key"""
        return f"embedding:v1:{hashlib.md5(text.encode()).hexdigest()}"

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """从Redis获取缓存"""
        key = self._get_cache_key(text)
        data = self.redis.get(key)
        if data:
            import pickle
            return pickle.loads(data)
        return None

    def _set_to_cache(self, text: str, embedding: List[float]) -> None:
        """存入Redis缓存"""
        key = self._get_cache_key(text)
        import pickle
        self.redis.setex(key, self.cache_ttl, pickle.dumps(embedding))

    async def embed(self, text: str) -> List[float]:
        """
        单条文本向量化

        Args:
            text: 输入文本

        Returns:
            1024维向量列表
        """
        # 先查缓存
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached

        # 缓存未命中，调用API
        result = await self._call_api([text])
        embedding = result[0]

        # 存入缓存
        self._set_to_cache(text, embedding)

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本向量化

        Args:
            texts: 文本列表，最多25条

        Returns:
            向量列表，每个元素是1024维向量
        """
        # 批量查缓存
        results: List[Optional[List[float]]] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # 对未命中的调用API
        if uncached_texts:
            new_embeddings = await self._call_api(uncached_texts)
            for idx, emb in zip(uncached_indices, new_embeddings):
                results[idx] = emb
                # 存入缓存
                self._set_to_cache(texts[idx], emb)

        return results

    async def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用百炼 embedding API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        params = {
            "model": self.model,
            "input": texts
        }

        response = await self.client.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=params
        )
        response.raise_for_status()
        result = response.json()
        # 打印embedding数量和维度（用于调试）
        if result.get("data"):
            dim = len(result["data"][0]["embedding"])
            logger.debug(f"Model: {self.model}, Dimension: {dim}")

        if "data" in result:
            embeddings = [item["embedding"] for item in sorted(result["data"], key=lambda x: x["index"])]
            return embeddings

        raise ValueError(f"Unexpected API response: {result}")


# 单例模式
_text_embedding: Optional[TextEmbedding] = None


def get_text_embedding() -> TextEmbedding:
    """获取文本向量化服务单例"""
    global _text_embedding
    if _text_embedding is None:
        _text_embedding = TextEmbedding()
    return _text_embedding
