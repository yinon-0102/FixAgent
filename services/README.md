# Services 模块

## 模块概述

Services 模块是系统的**核心服务层**，封装了与外部服务交互的所有客户端：
- **LLM Service**: 调用阿里云百炼DashScope API，执行大模型对话和生成
- **Vector Service**: 管理Redis向量库，实现向量存储和相似度检索
- **Graph Service**: 管理Neo4j图数据库，实现图谱查询和关系推理

本模块设计原则：
- **单一职责**: 每个服务类只负责一种外部服务的交互
- **接口统一**: 所有服务提供一致的调用接口
- **错误处理**: 统一异常处理和重试机制
- **可测试性**: 依赖注入，便于单元测试和Mock

## 服务列表

| 服务 | 文件 | 职责 | 外部依赖 |
|-----|------|------|---------|
| `llm_service` | llm_service.py | 大模型调用封装 | 阿里云百炼DashScope API |
| `vector_service` | vector_service.py | 向量数据库服务 | Redis Search |
| `graph_service` | graph_service.py | 图数据库服务 | Neo4j |

## 技术选型

| 组件 | 选型 | 理由 |
|-----|------|------|
| 大模型 | 阿里云百炼DashScope | 国产、Qwen系列强、API稳定 |
| 向量库 | Redis Search | 国产环境友好、高性能、内存数据库 |
| 图数据库 | Neo4j | 国产化支持好、Java生态完善、Cypher简洁 |

## 项目中的实现

### llm_service.py - 大模型服务

```python
# services/llm_service.py
"""
LLM服务模块

封装阿里云百炼DashScope API，提供对话和生成能力。
支持流式输出，适用于SSE场景。
"""

import json
import httpx
from typing import AsyncIterator, Optional, List, Dict, Any
from config.settings import get_settings


class LLMService:
    """
    大模型服务类

    封装阿里云百炼API，支持：
    - 同步/异步对话
    - 流式输出
    - 多轮对话上下文
    """

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.dashscope_api_key
        self.model = self.settings.llm_model
        self.api_base = "https://dashscope.aliyuncs.com/api/v1"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any] | AsyncIterator[str]:
        """
        对话接口

        Args:
            messages: 对话消息列表，格式：[{"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成长度
            stream: 是否流式输出

        Returns:
            非流式：完整响应字典
            流式：异步生成器yield每个token
        """
        params = {
            "model": self.model,
            "input": {"messages": messages},
            "parameters": {
                "temperature": temperature or self.settings.llm_temperature,
                "top_p": self.settings.llm_top_p,
                "max_tokens": max_tokens or self.settings.llm_max_tokens,
                "stream": stream
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            if stream:
                return self._stream_chat(client, headers, params)
            else:
                return await self._sync_chat(client, headers, params)

    async def _sync_chat(
        self,
        client: httpx.AsyncClient,
        headers: Dict,
        params: Dict
    ) -> Dict[str, Any]:
        """同步对话"""
        response = await client.post(
            f"{self.api_base}/services/a2t/text-generation/generation",
            headers=headers,
            json=params
        )
        response.raise_for_status()
        result = response.json()

        if "output" in result and "text" in result["output"]:
            return {
                "content": result["output"]["text"],
                "usage": result.get("usage", {}),
                "request_id": result.get("request_id")
            }
        return result

    async def _stream_chat(
        self,
        client: httpx.AsyncClient,
        headers: Dict,
        params: Dict
    ) -> AsyncIterator[str]:
        """流式对话，返回token生成器"""
        async with client.stream(
            "POST",
            f"{self.api_base}/services/a2t/text-generation/generation",
            headers=headers,
            json=params
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            data = json.loads(data_str)
                            if "output" in data and "text" in data["output"]:
                                token = data["output"]["text"]
                                yield token
                        except json.JSONDecodeError:
                            continue

    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        文本向量化

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        params = {
            "model": self.settings.embedding_model,
            "input": {"texts": texts}
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

            embeddings = []
            for item in result.get("output", {}).get("embeddings", []):
                embeddings.append(item["embedding"])

            return embeddings


# 单例模式
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """获取LLM服务单例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
```

### vector_service.py - 向量数据库服务

```python
# services/vector_service.py
"""
向量数据库服务模块

基于Redis Search实现向量存储和相似度检索。
支持：
- 向量添加/删除/更新
- 余弦相似度检索
- 带过滤条件的检索
- 批量操作
"""

import json
import redis
from typing import List, Optional, Dict, Any, Tuple
from config.settings import get_settings


class VectorService:
    """
    向量服务类

    基于Redis Search实现向量库功能：
    - 使用FT.CREATE创建向量索引
    - 使用FT.SEARCH进行向量检索
    - 使用HSET/HGET管理向量数据
    """

    def __init__(self):
        self.settings = get_settings()
        self.redis_client = redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            db=self.settings.redis_db,
            password=self.settings.redis_password,
            decode_responses=True
        )

        # 索引配置
        self.KNOWLEDGE_INDEX = "knowledge:vectors"
        self.IMAGE_INDEX = "image:vectors"

    def ensure_index(self, index_name: str, dimension: int = 1536):
        """
        确保向量索引存在，不存在则创建

        Args:
            index_name: 索引名称
            dimension: 向量维度
        """
        try:
            self.redis_client.execute_command(
                "FT.CREATE", index_name,
                "SCHEMA",
                "id", "TEXT",
                "content", "TEXT",
                "embedding", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", dimension,
                "DISTANCE_METRIC", "COSINE"
            )
        except redis.exceptions.ResponseError as e:
            if "already exists" not in str(e):
                raise e

    async def add_vector(
        self,
        index_name: str,
        doc_id: str,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加向量

        Args:
            index_name: 索引名称
            doc_id: 文档ID
            embedding: 向量
            content: 文本内容
            metadata: 元数据

        Returns:
            是否成功
        """
        key = f"{index_name}:{doc_id}"

        # 将向量转换为字节
        import struct
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        # 存储向量和内容
        pipe = self.redis_client.pipeline()
        pipe.hset(key, mapping={
            "id": doc_id,
            "content": content,
            "embedding": embedding_bytes
        })
        if metadata:
            pipe.hset(key, mapping={k: json.dumps(v) for k, v in metadata.items()})

        # 添加到索引
        pipe.execute()

        # 同步到索引
        self.redis_client.execute_command(
            "FT.SEARCH", index_name, f"@id:{{{doc_id}}}", "LIMIT", 0, 0
        )

        return True

    async def search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        向量检索

        Args:
            index_name: 索引名称
            query_vector: 查询向量
            top_k: 返回数量
            filters: 元数据过滤条件

        Returns:
            检索结果列表
        """
        # 构建查询
        import struct
        query_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

        # 构建搜索命令
        cmd = [
            "FT.SEARCH", index_name,
            f"*=>[KNN {top_k} @embedding $vec AS score]",
            "SORTBY", "score", "ASC",
            "RETURN", 3, "id", "content", "score",
            "PARAMS", "2", "vec", query_bytes,
            "LIMIT", 0, top_k
        ]

        try:
            results = self.redis_client.execute_command(*cmd)

            # 解析结果
            search_results = []
            if results and len(results) > 1:
                # 跳过头部元数据
                for i in range(1, len(results)):
                    if isinstance(results[i], list) and len(results[i]) >= 3:
                        search_results.append({
                            "id": results[i][1],  # id字段
                            "content": results[i][2],  # content字段
                            "score": float(results[i][3]) if len(results[i]) > 3 else 0.0
                        })

            return search_results

        except redis.exceptions.ResponseError as e:
            # 索引可能不存在，返回空结果
            if "no such index" in str(e).lower():
                return []
            raise e

    async def delete_vector(self, index_name: str, doc_id: str) -> bool:
        """删除向量"""
        key = f"{index_name}:{doc_id}"
        self.redis_client.delete(key)
        return True

    async def get_vector(self, index_name: str, doc_id: str) -> Optional[Dict]:
        """获取单个向量"""
        key = f"{index_name}:{doc_id}"
        data = self.redis_client.hgetall(key)
        if data:
            import struct
            embedding_bytes = data.get("embedding", b"")
            if embedding_bytes:
                embedding = list(struct.unpack(f"{len(embedding_bytes)//4}f", embedding_bytes))
                data["embedding"] = embedding
            return data
        return None


# 单例模式
_vector_service: Optional[VectorService] = None


def get_vector_service() -> VectorService:
    """获取向量服务单例"""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service
```

### graph_service.py - 图数据库服务

```python
# services/graph_service.py
"""
图数据库服务模块

基于Neo4j实现图谱存储和查询。
支持：
- 实体管理（创删改查）
- 关系管理
- 路径查询
- 图谱统计
"""

from typing import List, Optional, Dict, Any
from neo4j import AsyncGraphDatabase, AsyncDriver
from config.settings import get_settings


class GraphService:
    """
    图谱服务类

    基于Neo4j实现：
    - 节点CRUD
    - 关系CRUD
    - Cypher查询
    - 路径查询
    """

    def __init__(self):
        self.settings = get_settings()
        self._driver: Optional[AsyncDriver] = None

    async def get_driver(self) -> AsyncDriver:
        """获取或创建驱动连接"""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password)
            )
        return self._driver

    async def close(self):
        """关闭连接"""
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ==================== 节点操作 ====================

    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建节点

        Args:
            label: 节点标签
            properties: 节点属性

        Returns:
            创建的节点
        """
        driver = await self.get_driver()
        async with driver.session() as session:
            query = f"""
            CREATE (n:{label} $props)
            RETURN n
            """
            result = await session.run(query, props=properties)
            record = await result.single()
            return dict(record["n"])

    async def find_nodes(
        self,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        查询节点

        Args:
            label: 节点标签
            properties: 属性过滤条件
            limit: 返回数量限制

        Returns:
            节点列表
        """
        driver = await self.get_driver()
        async with driver.session() as session:
            if properties:
                where_clause = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])
                query = f"""
                MATCH (n:{label})
                WHERE {where_clause}
                RETURN n
                LIMIT $limit
                """
                params = {**properties, "limit": limit}
            else:
                query = f"""
                MATCH (n:{label})
                RETURN n
                LIMIT $limit
                """
                params = {"limit": limit}

            result = await session.run(query, params)
            nodes = []
            async for record in result:
                nodes.append(dict(record["n"]))
            return nodes

    async def delete_node(self, label: str, property_name: str, property_value: Any) -> bool:
        """删除节点"""
        driver = await self.get_driver()
        async with driver.session() as session:
            query = f"""
            MATCH (n:{label})
            WHERE n.{property_name} = $value
            DELETE n
            """
            await session.run(query, value=property_value)
            return True

    # ==================== 关系操作 ====================

    async def create_relation(
        self,
        source_label: str,
        source_property: str,
        source_value: Any,
        target_label: str,
        target_property: str,
        target_value: Any,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        创建关系

        Args:
            source_label: 源节点标签
            source_property: 源节点属性名
            source_value: 源节点属性值
            target_label: 目标节点标签
            target_property: 目标节点属性名
            target_value: 目标节点属性值
            relation_type: 关系类型
            properties: 关系属性

        Returns:
            是否成功
        """
        driver = await self.get_driver()
        async with driver.session() as session:
            query = f"""
            MATCH (s:{source_label}), (t:{target_label})
            WHERE s.{source_property} = $source_value
              AND t.{target_property} = $target_value
            CREATE (s)-[r:{relation_type} $props]->(t)
            RETURN r
            """
            props = properties or {}
            result = await session.run(
                query,
                source_value=source_value,
                target_value=target_value,
                props=props
            )
            record = await result.single()
            return record is not None

    async def find_relations(
        self,
        start_label: str,
        end_label: str,
        relation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        查询关系

        Args:
            start_label: 起始节点标签
            end_label: 终止节点标签
            relation_type: 关系类型过滤
            limit: 返回数量

        Returns:
            关系列表
        """
        driver = await self.get_driver()
        async with driver.session() as session:
            rel_pattern = f"[r:{relation_type}]" if relation_type else "[r]"
            query = f"""
            MATCH (s:{start_label})-{rel_pattern}-(t:{end_label})
            RETURN s, r, t
            LIMIT $limit
            """
            result = await session.run(query, limit=limit)

            relations = []
            async for record in result:
                relations.append({
                    "source": dict(record["s"]),
                    "relation": dict(record["r"]),
                    "target": dict(record["t"])
                })
            return relations

    # ==================== 图谱扩展查询 ====================

    async def expand_from_entity(
        self,
        entity_name: str,
        entity_label: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        从实体出发扩展查询相关实体

        Args:
            entity_name: 实体名称
            entity_label: 实体标签
            max_depth: 最大扩展深度

        Returns:
            扩展结果（节点和关系）
        """
        driver = await self.get_driver()
        async with driver.session() as session:
            query = f"""
            MATCH path = (start:{entity_label} {{name: $name}})-[*1..{max_depth}]-(related)
            RETURN path,
                   nodes(path) as node_list,
                   relationships(path) as rel_list
            LIMIT 50
            """
            result = await session.run(query, name=entity_name)

            nodes_set = {}
            relations_list = []

            async for record in result:
                for node in record["node_list"]:
                    node_id = id(node)
                    if node_id not in nodes_set:
                        nodes_set[node_id] = {
                            "id": node.get("name", str(node_id)),
                            "label": list(node.labels)[0] if node.labels else "Unknown",
                            "properties": dict(node)
                        }
                for rel in record["rel_list"]:
                    relations_list.append({
                        "source": dict(rel.start_node).get("name"),
                        "target": dict(rel.end_node).get("name"),
                        "type": rel.type,
                        "properties": dict(rel)
                    })

            return {
                "nodes": list(nodes_set.values()),
                "relations": relations_list
            }

    async def find_shortest_path(
        self,
        source_name: str,
        target_name: str,
        source_label: str = "Entity",
        target_label: str = "Entity"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        查找最短路径

        Args:
            source_name: 起点名称
            target_name: 终点名称
            source_label: 起点标签
            target_label: 终点标签

        Returns:
            路径节点列表，不存在则返回None
        """
        driver = await self.get_driver()
        async with driver.session() as session:
            query = f"""
            MATCH path = shortestPath(
                (s:{source_label} {{name: $source_name}})-[*]-(t:{target_label} {{name: $target_name}})
            )
            RETURN path
            """
            result = await session.run(query, source_name=source_name, target_name=target_name)
            record = await result.single()

            if record:
                path = record["path"]
                return [
                    {
                        "name": node.get("name"),
                        "label": list(node.labels)[0] if node.labels else "Unknown"
                    }
                    for node in path.nodes
                ]
            return None

    # ==================== 统计查询 ====================

    async def get_stats(self) -> Dict[str, Any]:
        """
        获取图谱统计信息

        Returns:
            统计信息
        """
        driver = await self.get_driver()
        async with driver.session() as session:
            # 统计各类型节点数量
            query = """
            MATCH (n)
            WITH labels(n)[0] as label, count(n) as count
            RETURN label, count
            """
            node_result = await session.run(query)
            node_counts = {record["label"]: record["count"] async for record in node_result}

            # 统计各类型关系数量
            query = """
            MATCH ()-[r]->()
            WITH type(r) as rel_type, count(r) as count
            RETURN rel_type, count
            """
            rel_result = await session.run(query)
            rel_counts = {record["rel_type"]: record["count"] async for record in rel_result}

            return {
                "total_nodes": sum(node_counts.values()),
                "total_relations": sum(rel_counts.values()),
                "node_types": node_counts,
                "relation_types": rel_counts
            }


# 单例模式
_graph_service: Optional[GraphService] = None


def get_graph_service() -> GraphService:
    """获取图谱服务单例"""
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service
```

## 使用示例

### 1. LLM服务调用

```python
# agents/orchestrator_agent.py
from services.llm_service import get_llm_service

llm = get_llm_service()

# 同步调用
result = await llm.chat(
    messages=[
        {"role": "system", "content": "你是一个专业的设备检修助手"},
        {"role": "user", "content": "电动机轴承过热是什么原因？"}
    ]
)
print(result["content"])

# 流式调用
async for token in llm.chat(messages=[...], stream=True):
    print(token, end="", flush=True)
```

### 2. 向量检索

```python
# tools/knowledge_retrieval_tool.py
from services.vector_service import get_vector_service

vector_service = get_vector_service()

# 添加向量
await vector_service.add_vector(
    index_name="knowledge:vectors",
    doc_id="kb_001",
    embedding=[0.1, 0.2, ...],  # 1536维
    content="电动机轴承过热的可能原因包括：1. 润滑不良...",
    metadata={"category": "motor", "tags": ["轴承", "过热"]}
)

# 检索
results = await vector_service.search(
    index_name="knowledge:vectors",
    query_vector=query_embedding,
    top_k=10
)
```

### 3. 图谱查询

```python
# agents/diagnosis_agent.py
from services.graph_service import get_graph_service

graph = get_graph_service()

# 从轴承过热扩展查询相关实体
related = await graph.expand_from_entity(
    entity_name="轴承过热",
    entity_label="Symptom",
    max_depth=2
)

# 查找故障原因到解决方案的路径
path = await graph.find_shortest_path(
    source_name="润滑不良",
    target_name="补充润滑",
    source_label="Cause",
    target_label="Solution"
)
```

## 与Java后端的对应关系

| Python Service | Java Service | 说明 |
|----------------|--------------|------|
| `LLMService` | `LLMService` | 调用百炼API |
| `VectorService` | `VectorService` | Redis向量检索 |
| `GraphService` | `GraphService` | Neo4j图查询 |

## 文件结构

```
services/
├── __init__.py
├── README.md                    # 本文件
├── llm_service.py              # 大模型服务
├── vector_service.py            # 向量数据库服务
└── graph_service.py            # 图数据库服务
```

## 注意事项

1. **连接池管理**: 所有服务使用单例模式，避免重复创建连接
2. **异步优先**: 使用async/await实现非阻塞IO
3. **错误处理**: 统一捕获外部服务异常，转换为内部异常
4. **资源释放**: 使用上下文管理器确保资源正确释放
5. **超时控制**: 所有HTTP/数据库调用都设置合理的超时时间
