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
        self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

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
            "messages": messages,
            "temperature": temperature or self.settings.llm_temperature,
            "top_p": self.settings.llm_top_p,
            "max_tokens": max_tokens or self.settings.llm_max_tokens
        }

        if stream:
            params["stream"] = True

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if stream:
            return self._stream_chat(self.client, headers, params)
        else:
            return await self._sync_chat(self.client, headers, params)

    async def _sync_chat(
        self,
        client: httpx.AsyncClient,
        headers: Dict,
        params: Dict
    ) -> Dict[str, Any]:
        """同步对话"""
        response = await client.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=params
        )
        response.raise_for_status()
        result = response.json()

        if "choices" in result:
            return {
                "content": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "request_id": result.get("id")
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
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=params
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                token = data["choices"][0]["delta"].get("content", "")
                                if token:
                                    yield token
                        except json.JSONDecodeError:
                            continue

# 单例模式
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """获取LLM服务单例"""
    """保证全局只有一个 LLMService 实例,连接资源复用"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service