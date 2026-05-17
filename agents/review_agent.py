"""
输出审核 Agent（ReviewAgent）

纯 LLM 校验，无工具。对 FixAgent 的输出做最终质量检查。
基于 Reflexion 思想：生成后反思，发现问题则修正。

【校验维度】
- 诊断依据是否充分（有无凭空推测，结论是否有检索证据支撑）
- 安全提示是否完整（高压/高温/化学品/重物等操作）
- 有无遗漏的可能故障原因
- 回答内容有无自相矛盾

【调用链】
api/main.py → FixAgent 完成推理 → ReviewAgent.review() → 审核/修正 → 返回最终结果

【关联】
- 继承 BaseAgent，使用 run() 单次 LLM 调用
- 上游：FixAgent 的 AgentOutput（含 message + react_trace）
- 无工具，纯 LLM 推理
"""

import json
import time
import logging
from typing import Optional, Dict, Any

from agents.base_agent import BaseAgent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


REVIEW_SYSTEM_PROMPT = """你是设备检修回答的质量审核专家。你的任务是审核AI助手的回答质量，确保回答准确、安全、完整。

## 审核维度

### 1. 诊断依据充分性
- 诊断结论是否有知识库检索结果或图谱查询结果支撑？
- 是否存在凭空推测的技术细节（如编造数据、虚构零件型号）？
- 如果工具返回结果为空或不相关，回答是否坦诚说明信息不足？

### 2. 安全提示完整性
- 涉及高压电气操作时，是否提醒断电和验电？
- 涉及高温部件时，是否提醒冷却等待和防烫伤？
- 涉及化学品（冷却液、润滑油等）时，是否提醒防护措施？
- 涉及重物吊装时，是否提醒使用合适工具和人员配合？
- 是否遗漏了其他明显的安全风险？

### 3. 故障原因完整性
- 是否遗漏了常见的可能故障原因？
- 原因排列是否按可能性或严重程度合理排序？

### 4. 内容一致性
- 回答中是否存在前后矛盾的内容？
- 维修步骤是否逻辑连贯？

## 输出格式

严格按以下 JSON 格式输出，不要输出其他内容：

```json
{
  "status": "approved 或 revised",
  "issues": ["发现的问题描述列表，approved时为空数组"],
  "revised_content": "修正后的完整回答内容。approved时为空字符串"
}
```

- **approved**：回答质量合格，无需修改
- **revised**：发现问题，revised_content 中给出修正后的完整回答

## 审核原则

- 不要过度审核：小瑕疵不影响使用的，判为 approved
- 只在确实存在安全隐患、事实错误或重大遗漏时才 revised
- 修正时保持原回答的风格和结构，只修复具体问题
- 闲聊类回答（不涉及技术诊断）直接 approved
"""


class ReviewAgent(BaseAgent):
    """
    输出审核 Agent

    纯 LLM 调用，无工具。对 FixAgent 输出做最终校验。
    发现问题时返回修正后的内容，无问题时返回 approved。
    """

    @property
    def name(self) -> str:
        return "review_agent"

    @property
    def description(self) -> str:
        return "输出审核Agent：校验回答的准确性、安全性和完整性"

    def get_system_prompt(self) -> str:
        return REVIEW_SYSTEM_PROMPT

    async def review(self, fix_output: AgentOutput) -> AgentOutput:
        """
        审核 FixAgent 的输出

        Args:
            fix_output: FixAgent 的 AgentOutput

        Returns:
            AgentOutput，message 为最终回答（原样或修正后），
            metadata 中包含 review_status 和 review_issues
        """
        start_time = time.time()

        fix_message = fix_output.message
        react_trace = fix_output.metadata.get("react_trace", [])

        trace_summary = self._summarize_trace(react_trace)

        user_content = f"""## AI助手的回答

{fix_message}

## 工具调用记录

{trace_summary}

请审核以上回答的质量。"""

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_content}
        ]

        try:
            response = await self.llm_service.chat(
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.get("content", "")
            review_result = self._parse_review(content)
        except Exception as e:
            logger.warning(f"ReviewAgent failed, passing through original: {e}")
            review_result = {
                "status": "approved",
                "issues": [],
                "revised_content": "",
                "error": str(e)
            }

        latency_ms = int((time.time() - start_time) * 1000)

        if review_result["status"] == "revised" and review_result.get("revised_content"):
            final_message = review_result["revised_content"]
        else:
            final_message = fix_message

        merged_metadata = {**fix_output.metadata}
        merged_metadata["review_status"] = review_result["status"]
        merged_metadata["review_issues"] = review_result.get("issues", [])
        merged_metadata["review_latency_ms"] = latency_ms
        merged_metadata["total_latency_ms"] = fix_output.latency_ms + latency_ms

        return AgentOutput(
            agent_name="fix_agent",
            message=final_message,
            intention=fix_output.intention,
            tools_used=fix_output.tools_used,
            metadata=merged_metadata,
            latency_ms=fix_output.latency_ms + latency_ms,
            raw_response=fix_output.raw_response
        )

    def _summarize_trace(self, react_trace: list) -> str:
        """将 react_trace 转为可读文本，供审核 LLM 理解工具调用过程"""
        if not react_trace:
            return "无工具调用记录（直接回答）"

        lines = []
        for step in react_trace:
            iteration = step.get("iteration", "?")
            action = step.get("action", "unknown")

            if action == "tool_call":
                for tc in step.get("tool_calls", []):
                    name = tc.get("name", "unknown")
                    args = tc.get("arguments", {})
                    result_summary = tc.get("result_summary", "")[:300]
                    lines.append(
                        f"[步骤{iteration}] 调用工具 {name}，参数: {json.dumps(args, ensure_ascii=False)}"
                    )
                    lines.append(f"  → 结果: {result_summary}")
            elif action == "finish":
                lines.append(f"[步骤{iteration}] 生成最终回答")

        return "\n".join(lines) if lines else "无工具调用记录"

    def _parse_review(self, content: str) -> Dict[str, Any]:
        """解析审核 LLM 返回的 JSON"""
        try:
            import re
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)
            return {
                "status": data.get("status", "approved"),
                "issues": data.get("issues", []),
                "revised_content": data.get("revised_content", "")
            }
        except (json.JSONDecodeError, ValueError, AttributeError):
            logger.warning(f"ReviewAgent JSON parse failed, defaulting to approved: {content[:100]}")
            return {
                "status": "approved",
                "issues": [],
                "revised_content": ""
            }


# 单例
_review_agent = None


def get_review_agent() -> ReviewAgent:
    global _review_agent
    if _review_agent is None:
        from services.llm_service import get_llm_service
        _review_agent = ReviewAgent(get_llm_service())
    return _review_agent
