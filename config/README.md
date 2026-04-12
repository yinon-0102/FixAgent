# Config 模块

## 模块概述

Config 模块是整个项目的配置管理中心，负责管理所有配置信息，包括：
- API密钥（DashScope百炼API）
- 数据库连接（Redis/MySQL/Neo4j）
- 模型配置（LLM模型、Embedding模型）
- 服务配置（主机、端口、超时等）

本模块采用**中心化配置**思想，所有配置集中管理，避免硬编码，便于部署和切换环境。

## 配置项详解

| 配置项 | 环境变量 | 类型 | 默认值 | 说明 |
|-------|---------|------|--------|------|
| DashScope API Key | `DASHSCOPE_API_KEY` | str | **必需** | 阿里云百炼API密钥 |
| Redis 主机 | `REDIS_HOST` | str | localhost | Redis服务器地址 |
| Redis 端口 | `REDIS_PORT` | int | 6379 | Redis服务端口 |
| LLM 模型 | `LLM_MODEL` | str | qwen-plus | 阿里云百炼模型名称 |
| Embedding模型 | `EMBEDDING_MODEL` | str | text-embedding-v4 | 向量化模型 |
| 日志级别 | `LOG_LEVEL` | str | INFO | DEBUG/INFO/WARNING/ERROR |
| SSE超时 | `SSE_TIMEOUT` | int | 300 | SSE连接超时时间(秒) |

## 技术选型

| 组件 | 选型 | 理由 |
|-----|------|------|
| 配置管理 | python-dotenv | 加载.env文件，環境隔离 |
| 配置读取 | pydantic-settings | 类型安全、自动验证、支持嵌套配置 |

## 项目中的实现

### settings.py 完整实现

```python
# config/settings.py
"""
FixAgent 配置管理模块

所有配置通过环境变量或 .env 文件加载，确保敏感信息不硬编码。
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    应用配置类

    使用pydantic-settings自动从环境变量或.env文件加载配置。
    配置项会在启动时进行验证，缺少必需项会抛出异常。
    """

    # ==================== API Keys ====================
    dashscope_api_key: str

    # ==================== Redis 配置 ====================
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    @property
    def redis_url(self) -> str:
        """构建Redis连接URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ==================== Neo4j 配置 ====================
    neo4j_host: str = "localhost"
    neo4j_port: int = 7687
    neo4j_username: str = "neo4j"
    neo4j_password: str = "neo4j"
    neo4j_database: str = "neo4j"

    @property
    def neo4j_uri(self) -> str:
        """构建Neo4j连接URI"""
        return f"bolt://{self.neo4j_host}:{self.neo4j_port}"

    # ==================== MySQL 配置 ====================
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "fixagent"
    mysql_username: str = "root"
    mysql_password: str = ""

    @property
    def mysql_url(self) -> str:
        """构建MySQL连接URL"""
        return f"mysql+pymysql://{self.mysql_username}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"

    # ==================== LLM 模型配置 ====================
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    llm_top_p: float = 0.8

    # ==================== Embedding 配置 ====================
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1536  # text-embedding-v4 输出维度

    # ==================== FastAPI 配置 ====================
    api_host: str = "0.0.0.0"
    api_port: int = 8001

    # ==================== SSE 配置 ====================
    sse_timeout: int = 300  # SSE连接超时时间（秒）
    sse_heartbeat_interval: int = 30  # 心跳间隔（秒）

    # ==================== Python工具服务配置 ====================
    python_tools_url: str = "http://localhost:8001"
    tools_timeout: int = 30  # 工具调用超时（秒）

    # ==================== 日志配置 ====================
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ==================== CORS 配置 ====================
    cors_origins: list = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # 环境变量不区分大小写


@lru_cache()  # 单例模式，避免重复加载
def get_settings() -> Settings:
    """
    获取配置单例

    使用@lru_cache确保整个应用生命周期内只加载一次配置。
    """
    return Settings()
```

### .env 文件示例

```bash
# .env 示例文件

# ==================== API Keys ====================
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# ==================== Redis ====================
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# ==================== Neo4j ====================
NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j123

# ==================== MySQL ====================
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=fixagent
MYSQL_USERNAME=root
MYSQL_PASSWORD=root123

# ==================== 模型配置 ====================
LLM_MODEL=qwen-plus
LLM_TEMPERATURE=0.7
EMBEDDING_MODEL=text-embedding-v4

# ==================== Python工具服务 ====================
PYTHON_TOOLS_URL=http://localhost:8001
TOOLS_TIMEOUT=30

# ==================== 日志 ====================
LOG_LEVEL=INFO
```

### 使用示例

```python
# 在其他模块中使用配置
from config.settings import get_settings

settings = get_settings()

# 调用LLM服务
response = call_dashscope(
    api_key=settings.dashscope_api_key,
    model=settings.llm_model,
    messages=[...]
)

# 连接Redis
redis_client = redis.from_url(settings.redis_url)
```

## 与Java后端的配置对应关系

| Python配置 | Java配置 (application.yml) | 说明 |
|-----------|---------------------------|------|
| `DASHSCOPE_API_KEY` | `dashscope.api-key` | 百炼API密钥 |
| `REDIS_HOST` | `spring.data.redis.host` | Redis连接 |
| `NEO4J_HOST` | `neo4j.uri` | Neo4j连接 |
| `MYSQL_*` | `spring.datasource.*` | MySQL连接 |

## 文件结构

```
config/
├── __init__.py
├── README.md                    # 本文件
└── settings.py                  # 配置类定义
```

## 注意事项

1. **敏感信息保护**: `.env`文件不应提交到版本控制，需添加到`.gitignore`
2. **配置验证**: pydantic会在启动时验证配置，缺少必需项会立即报错
3. **环境切换**: 通过`SPRING_PROFILES_ACTIVE`(Java)或`ENV`(Python)切换不同环境配置
4. **配置热更新**: 生产环境建议使用配置中心（如Apollo、Nacos）实现配置热更新
