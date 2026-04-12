# Embeddings 模块

## 模块概述

Embeddings 模块负责将**文本**和**图像**转换为向量表示，是实现跨模态检索的基石。本模块提供：
- **文本向量化**: 调用阿里云百炼text-embedding-v4 API
- **图像向量化**: 使用OpenAI CLIP模型实现图像编码
- **多模态融合**: 文本和图像映射到统一向量空间

核心价值：用户可以用文本查询相似图片，也可以用图片检索相关文本，实现真正的跨模态语义检索。

## 向量模型

| 类型 | 模型 | 来源 | 维度 | 特点 |
|-----|------|------|------|------|
| 文本向量 | text-embedding-v4 | 阿里云百炼 | 1536 | 国产、中文优化、托管服务 |
| 图像向量 | CLIP (ViT-B/32) | OpenAI | 512 | 跨模态、统一空间 |
| 多模态向量 | CLIP | OpenAI | 512 | 文本/图像共用 |

## 技术选型

| 组件 | 选型 | 理由 |
|-----|------|------|
| 文本嵌入 | 阿里云百炼text-embedding-v4 | 国产、中文支持好、API稳定 |
| 图像嵌入 | CLIP (ViT-B/32) | 跨模态能力强、开源可用 |
| 向量库 | Redis | 国产环境友好、高性能 |

## 项目中的实现

### text_embedding.py - 文本向量化

```python
# embeddings/text_embedding.py
"""
文本向量化模块

使用阿里云百炼text-embedding-v4将文本转换为向量。
支持批量处理和缓存。
"""

import httpx
from typing import List, Optional, Dict, Any
import numpy as np
from config.settings import get_settings


class TextEmbedding:
    """
    文本向量化类

    封装阿里云百炼文本嵌入API，提供：
    - 单文本/批量文本向量化
    - 向量归一化
    - 缓存支持
    """

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.dashscope_api_key
        self.model = self.settings.embedding_model
        self.dimension = self.settings.embedding_dim  # 1536
        self.api_base = "https://dashscope.aliyuncs.com/api/v1"

        # 简单内存缓存
        self._cache: Dict[str, List[float]] = {}

    def _get_cache_key(self, text: str) -> str:
        """生成缓存key"""
        return f"text:{hash(text)}"

    def _normalize(self, vector: List[float]) -> List[float]:
        """L2归一化"""
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    async def embed_single(self, text: str, use_cache: bool = True) -> List[float]:
        """
        单文本向量化

        Args:
            text: 输入文本
            use_cache: 是否使用缓存

        Returns:
            归一化后的向量
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]

        params = {
            "model": self.model,
            "input": {"texts": [text]}
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.api_base}/services/a2t/text-embedding/embedding",
                headers=headers,
                json=params
            )
            response.raise_for_status()
            result = response.json()

            # 解析返回的向量
            embeddings = result.get("output", {}).get("embeddings", [])
            if embeddings:
                embedding = embeddings[0]["embedding"]
                normalized = self._normalize(embedding)

                if use_cache:
                    self._cache[cache_key] = normalized

                return normalized

            raise ValueError(f"Failed to get embedding: {result}")

    async def embed_batch(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        批量文本向量化

        Args:
            texts: 文本列表
            use_cache: 是否使用缓存

        Returns:
            向量列表
        """
        results = []
        uncached_texts = []
        uncached_indices = []

        # 检查缓存
        for i, text in enumerate(texts):
            if use_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    results.append(self._cache[cache_key])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 批量请求未缓存的文本
        if uncached_texts:
            params = {
                "model": self.model,
                "input": {"texts": uncached_texts}
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/services/a2t/text-embedding/embedding",
                    headers=headers,
                    json=params
                )
                response.raise_for_status()
                result = response.json()

                embeddings = result.get("output", {}).get("embeddings", [])

                for i, emb in enumerate(embeddings):
                    normalized = self._normalize(emb["embedding"])
                    results.append(normalized)

                    if use_cache:
                        cache_key = self._get_cache_key(uncached_texts[i])
                        self._cache[cache_key] = normalized

        # 确保结果顺序与输入一致
        return results

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


# 单例模式
_text_embedding: Optional[TextEmbedding] = None


def get_text_embedding() -> TextEmbedding:
    """获取文本向量化实例"""
    global _text_embedding
    if _text_embedding is None:
        _text_embedding = TextEmbedding()
    return _text_embedding
```

### image_embedding.py - 图像向量化

```python
# embeddings/image_embedding.py
"""
图像向量化模块

使用CLIP模型将图像转换为向量。
支持：
- 图片URL/本地路径
- 批量处理
- 缓存
"""

import torch
import clip
from PIL import Image
import httpx
from io import BytesIO
from typing import List, Optional, Union
import numpy as np
from config.settings import get_settings


class ImageEmbedding:
    """
    图像向量化类

    使用CLIP模型将图像编码为向量：
    - 支持URL和本地路径
    - 自动下载和预处理
    - GPU加速（如果有）
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        self.model_name = model_name

        # 加载CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # 设置
        self.dimension = 512  # CLIP ViT-B/32 输出维度

    def _download_image(self, url: str) -> Image.Image:
        """从URL下载图片"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """加载图片"""
        if isinstance(image_source, Image.Image):
            return image_source
        elif image_source.startswith("http://") or image_source.startswith("https://"):
            # 同步方式使用httpx
            return self._download_image_sync(image_source)
        else:
            # 本地路径
            return Image.open(image_source).convert("RGB")

    def _download_image_sync(self, url: str) -> Image.Image:
        """同步方式下载图片"""
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L2归一化"""
        embeddings = embeddings.float()
        norms = embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings / norms

    async def embed_single(
        self,
        image_source: Union[str, Image.Image],
        use_cache: bool = False
    ) -> List[float]:
        """
        单图像向量化

        Args:
            image_source: 图片URL或本地路径
            use_cache: 是否使用缓存

        Returns:
            归一化后的向量
        """
        # 加载图片
        image = self._load_image(image_source)

        # 预处理
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # 编码
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = self._normalize(image_features)

        return image_features.cpu().numpy()[0].tolist()

    async def embed_batch(
        self,
        image_sources: List[Union[str, Image.Image]]
    ) -> List[List[float]]:
        """
        批量图像向量化

        Args:
            image_sources: 图片列表

        Returns:
            向量列表
        """
        images = [self._load_image(src) for src in image_sources]

        # 预处理批量图片
        image_inputs = torch.stack(
            [self.preprocess(img) for img in images]
        ).to(self.device)

        # 批量编码
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            image_features = self._normalize(image_features)

        return image_features.cpu().numpy().tolist()


# 单例模式
_image_embedding: Optional[ImageEmbedding] = None


def get_image_embedding() -> ImageEmbedding:
    """获取图像向量化实例"""
    global _image_embedding
    if _image_embedding is None:
        _image_embedding = ImageEmbedding()
    return _image_embedding
```

### multimodal_embedding.py - 多模态统一向量

```python
# embeddings/multimodal_embedding.py
"""
多模态统一向量模块

使用CLIP实现文本和图像到同一向量空间的映射，
支持真正的跨模态检索：
- 文本 ↔ 文本
- 图像 ↔ 图像
- 文本 ↔ 图像
"""

import torch
import clip
from PIL import Image
from typing import List, Optional, Union, Dict, Any
import numpy as np
from config.settings import get_settings

from .text_embedding import get_text_embedding
from .image_embedding import get_image_embedding


class MultimodalEmbedding:
    """
    多模态统一向量类

    使用CLIP将文本和图像映射到同一向量空间：
    - 文本编码: CLIP text encoder
    - 图像编码: CLIP image encoder
    - 统一归一化，确保可比较性
    """

    def __init__(self):
        # 复用单独的embedding服务
        self.text_embedder = get_text_embedding()
        self.image_embedder = get_image_embedding()

    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        文本向量化（使用CLIP）

        Args:
            text: 输入文本
            use_cache: 是否使用缓存

        Returns:
            512维归一化向量
        """
        return await self.text_embedder.embed_single(text, use_cache)

    async def embed_image(self, image_source: Union[str, Image.Image]) -> List[float]:
        """
        图像向量化

        Args:
            image_source: 图片URL或本地路径

        Returns:
            512维归一化向量
        """
        return await self.image_embedder.embed_single(image_source)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本向量化

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        return await self.text_embedder.embed_batch(texts)

    async def embed_images(self, image_sources: List[Union[str, Image.Image]]) -> List[List[float]]:
        """
        批量图像向量化

        Args:
            image_sources: 图片列表

        Returns:
            向量列表
        """
        return await self.image_embedder.embed_batch(image_sources)

    async def embed_query(
        self,
        query_text: Optional[str] = None,
        query_images: Optional[List[str]] = None
    ) -> List[float]:
        """
        混合查询向量化

        支持：
        - 纯文本查询
        - 纯图片查询
        - 图文混合查询（向量融合）

        Args:
            query_text: 查询文本
            query_images: 查询图片列表

        Returns:
            融合后的查询向量
        """
        query_vectors = []

        # 文本向量
        if query_text:
            text_vec = await self.embed_text(query_text)
            query_vectors.append(text_vec)

        # 图片向量
        if query_images:
            image_vecs = await self.embed_images(query_images)
            query_vectors.extend(image_vecs)

        if not query_vectors:
            raise ValueError("至少需要提供文本或图片查询")

        # 多向量取平均融合
        if len(query_vectors) == 1:
            return query_vectors[0]

        # 加权平均（文本权重稍高）
        weights = [0.6] if query_text else []
        weights.extend([0.4 / len(query_images)] * len(query_images))
        weights = weights[:len(query_vectors)]

        fused = np.average(query_vectors, axis=0, weights=weights).tolist()

        # 重新归一化
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = (np.array(fused) / norm).tolist()

        return fused

    def compute_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相似度分数 [-1, 1]
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return float(np.dot(vec1, vec2))

    async def search_similar(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        candidate_vectors: List[List[float]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        在候选集中搜索相似项

        Args:
            query: 查询文本
            candidates: 候选项列表（包含id、content等）
            candidate_vectors: 候选项对应的向量
            top_k: 返回数量

        Returns:
            按相似度排序的结果
        """
        # 查询向量
        query_vec = await self.embed_text(query)

        # 计算相似度
        results = []
        for i, candidate_vec in enumerate(candidate_vectors):
            similarity = self.compute_similarity(query_vec, candidate_vec)
            candidate = candidates[i].copy()
            candidate["score"] = similarity
            results.append(candidate)

        # 排序
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]


# 单例模式
_multimodal_embedding: Optional[MultimodalEmbedding] = None


def get_multimodal_embedding() -> MultimodalEmbedding:
    """获取多模态向量化实例"""
    global _multimodal_embedding
    if _multimodal_embedding is None:
        _multimodal_embedding = MultimodalEmbedding()
    return _multimodal_embedding
```

## 使用示例

### 1. 纯文本检索

```python
from embeddings import get_text_embedding

text_embedder = get_text_embedding()

# 单文本向量化
vec = await text_embedder.embed_single("电动机轴承故障")
print(f"向量维度: {len(vec)}")

# 批量向量化
texts = ["轴承过热", "润滑不良", "轴承磨损"]
vectors = await text_embedder.embed_batch(texts)
```

### 2. 纯图像检索

```python
from embeddings import get_image_embedding

image_embedder = get_image_embedding()

# 单图像向量化
vec = await image_embedder.embed_single("https://example.com/fault.jpg")

# 批量向量化
vecs = await image_embedder.embed_batch([
    "images/fault1.jpg",
    "images/fault2.jpg"
])
```

### 3. 跨模态检索

```python
from embeddings import get_multimodal_embedding

multi = get_multimodal_embedding()

# 用文本查询相似图片
query_vec = await multi.embed_query(query_text="轴承磨损")
image_vecs = await multi.embed_images(image_urls)

# 计算相似度
for i, img_vec in enumerate(image_vecs):
    similarity = multi.compute_similarity(query_vec, img_vec)
    print(f"图片{i}相似度: {similarity:.4f}")
```

### 4. 图文混合检索

```python
# 用户上传故障图片并描述现象
query_vec = await multi.embed_query(
    query_text="轴承区域异常",
    query_images=["user_upload/fault.jpg"]
)

# 在知识库中检索
results = await vector_service.search(
    index_name="knowledge:vectors",
    query_vector=query_vec,
    top_k=10
)
```

## 与Java后端的对应关系

| Python Embedding | Java实现 |
|-----------------|----------|
| `TextEmbedding` | `TextEmbeddingService` (调用百炼API) |
| `ImageEmbedding` | Python Tool (CLIP) → Java代理调用 |
| `MultimodalEmbedding` | `MultimodalEmbeddingService` |

## 文件结构

```
embeddings/
├── __init__.py
├── README.md                    # 本文件
├── text_embedding.py            # 文本向量化（百炼API）
├── image_embedding.py           # 图像向量化（CLIP）
└── multimodal_embedding.py      # 多模态统一向量
```

## 注意事项

1. **向量维度匹配**:
   - 百炼text-embedding-v4: 1536维
   - CLIP ViT-B/32: 512维
   - 存储和检索时需注意维度一致

2. **CLIP模型加载**:
   - 首次调用会下载模型（约340MB）
   - 建议在服务启动时预加载

3. **GPU加速**:
   - 检测到CUDA时会自动使用GPU
   - 无GPU时使用CPU，速度较慢

4. **缓存策略**:
   - 文本嵌入使用内存缓存（适合单实例）
   - 分布式部署时需使用Redis缓存

5. **向量归一化**:
   - 所有向量都会L2归一化
   - 确保余弦相似度计算准确
