"""
输出审核 Agent（3层确定性校验）

对 FixAgent 的输出进行确定性校验，零 LLM 调用。

【3层校验】
1. Grounding 校验 — 检查回答中的关键陈述是否有检索依据支撑
2. Graph 校验 — 检查回答中提到的图谱路径（设备→部件→故障→方案）是否在图谱证据中出现
3. Safety 校验 — 检查维修建议是否遗漏安全提醒（断电、防护装备、资质要求等）

【设计原则】
- 不调 LLM，纯规则 + 字符串匹配，延迟 < 300ms
- 不修改原回答内容，只添加校验标注
- 校验结果通过 metadata 返回，由流式接口决定如何展示
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional

from .base_agent import AgentOutput

logger = logging.getLogger(__name__)

# ==================== 安全关键词 ====================
SAFETY_KEYWORDS = [
    "断电", "停机", "断开电源", "切断电源",
    "防护", "护目镜", "手套", "安全帽", "防护装备",
    "资质", "持证", "专业人员", "授权人员",
    "高压", "带电", "漏电",
    "通风", "有毒", "气体",
]

SAFETY_SCENARIOS = {
    "电气": ["断电", "断开电源", "切断电源", "带电", "漏电"],
    "高压": ["断电", "高压", "专业人员", "资质"],
    "机械": ["停机", "防护"],
    "化学": ["通风", "防护", "手套"],
}


class ReviewAgent:
    """
    输出审核 Agent

    对 FixAgent 的回答进行 3 层确定性校验：
    1. Grounding（检索依据校验）
    2. Graph（图谱路径校验）
    3. Safety（安全规则引擎）

    零 LLM 调用，纯规则匹配，延迟 < 300ms。
    """

    @property
    def name(self) -> str:
        return "review_agent"

    @property
    def description(self) -> str:
        return "输出审核Agent：3层确定性校验（检索依据 + 图谱路径 + 安全规则）"

    async def review(self, fix_output: AgentOutput) -> AgentOutput:
        """
        对 FixAgent 输出进行 3 层校验

        Args:
            fix_output: FixAgent 的输出结果

        Returns:
            添加了校验信息的 AgentOutput
        """
        start_time = time.time()

        message = fix_output.message
        metadata = dict(fix_output.metadata)
        react_trace = metadata.get("react_trace", [])

        # 1. Grounding 校验
        grounding_result = self._check_grounding(message, react_trace)

        # 2. Graph 校验
        graph_result = self._check_graph(message, react_trace)

        # 3. Safety 校验
        safety_result = self._check_safety(message)

        # 汇总
        verification = {
            "grounding": grounding_result,
            "graph": graph_result,
            "safety": safety_result,
        }

        has_issues = (
            grounding_result.get("unverified_count", 0) > 0
            or graph_result.get("unverified_count", 0) > 0
            or safety_result.get("missing_count", 0) > 0
        )

        # 如果有安全遗漏，在回答末尾追加安全提醒
        final_message = message
        if safety_result.get("missing_count", 0) > 0:
            missing = safety_result.get("missing_warnings", [])
            safety_appendix = "\n\n⚠️ **安全提醒**：" + "；".join(missing) + "。"
            final_message = message + safety_appendix

        review_ms = int((time.time() - start_time) * 1000)

        metadata["verification"] = verification
        metadata["verification_has_issues"] = has_issues

        return AgentOutput(
            agent_name=self.name,
            message=final_message,
            intention=fix_output.intention,
            tools_used=fix_output.tools_used,
            metadata=metadata,
            latency_ms=fix_output.latency_ms + review_ms,
            raw_response=fix_output.raw_response,
        )

    def get_inline_markers(
        self, message: str, verification: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        根据校验结果生成内联标记位置

        用于流式输出时在未验证内容前插入 marker 事件。

        Args:
            message: 原始回答文本
            verification: 校验结果字典

        Returns:
            标记列表，每个元素 {"char_pos": int, "text": str, "type": str}
        """
        markers = []

        # 安全遗漏标记放在末尾
        safety = verification.get("safety", {})
        if safety.get("missing_count", 0) > 0:
            markers.append({
                "char_pos": len(message),
                "text": "以下为系统自动补充的安全提醒",
                "type": "safety_append",
            })

        # Grounding 未验证语句标记
        grounding = verification.get("grounding", {})
        for item in grounding.get("unverified_statements", []):
            pos = message.find(item.get("text", ""))
            if pos >= 0:
                markers.append({
                    "char_pos": pos,
                    "text": "此陈述未找到检索依据",
                    "type": "grounding_unverified",
                })

        markers.sort(key=lambda m: m["char_pos"])
        return markers

    # ==================== 内部校验方法 ====================

    def _check_grounding(
        self, message: str, react_trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Grounding 校验：检查回答是否有检索依据支撑

        简单策略：如果 FixAgent 使用了检索工具（知识库/图谱），
        则认为有依据；否则标记整段回答为"无检索依据"。
        """
        retrieval_tools = {
            "knowledge_search", "graph_query", "graph_image_search",
            "vector_search", "graph_diagnosis",
        }

        used_tools = set()
        for step in react_trace:
            tool_name = step.get("tool", "")
            if tool_name:
                used_tools.add(tool_name)

        has_retrieval = bool(used_tools & retrieval_tools)

        if has_retrieval:
            return {"verified": True, "unverified_count": 0, "unverified_statements": []}

        # 没有检索工具 → 如果回答较长，标记为未验证
        if len(message) > 100:
            return {
                "verified": False,
                "unverified_count": 1,
                "unverified_statements": [
                    {"text": message[:80], "reason": "回答未使用检索工具，缺少依据支撑"}
                ],
            }

        return {"verified": True, "unverified_count": 0, "unverified_statements": []}

    def _check_graph(
        self, message: str, react_trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Graph 校验：检查提到的图谱实体是否在工具返回中出现

        简单策略：目前仅检查是否使用了图谱工具，
        后续可扩展为实体级别的交叉验证。
        """
        graph_tools = {"graph_query", "graph_diagnosis", "graph_image_search"}

        used_tools = set()
        for step in react_trace:
            tool_name = step.get("tool", "")
            if tool_name:
                used_tools.add(tool_name)

        has_graph = bool(used_tools & graph_tools)

        if has_graph or not any(kw in message for kw in ["部件", "故障", "解决方案", "设备"]):
            return {"verified": True, "unverified_count": 0}

        return {
            "verified": False,
            "unverified_count": 1,
            "note": "回答中提及图谱实体但未使用图谱工具验证",
        }

    def _check_safety(self, message: str) -> Dict[str, Any]:
        """
        Safety 校验：检查维修建议是否遗漏安全提醒

        规则：如果回答涉及维修操作（含有操作性动词），
        检查是否包含相应安全关键词。
        """
        action_patterns = [
            r"更换", r"拆卸", r"安装", r"维修", r"检修",
            r"清洗", r"调整", r"校准", r"焊接", r"接线",
        ]

        has_action = any(re.search(p, message) for p in action_patterns)
        if not has_action:
            return {"missing_count": 0, "missing_warnings": []}

        # 检测涉及的场景
        missing_warnings = []

        # 电气相关
        electrical_keywords = ["电机", "电路", "电源", "线路", "接线", "配电", "变频"]
        if any(kw in message for kw in electrical_keywords):
            if not any(kw in message for kw in ["断电", "断开电源", "切断电源"]):
                missing_warnings.append("操作前请确保已断电")

        # 通用安全
        if not any(kw in message for kw in ["防护", "护目镜", "手套", "安全帽"]):
            if any(kw in message for kw in ["焊接", "切割", "打磨"]):
                missing_warnings.append("请佩戴相应防护装备")

        # 资质要求
        high_risk_keywords = ["高压", "特种", "压力容器", "起重"]
        if any(kw in message for kw in high_risk_keywords):
            if not any(kw in message for kw in ["资质", "持证", "专业人员", "授权"]):
                missing_warnings.append("该操作需要持证专业人员执行")

        return {
            "missing_count": len(missing_warnings),
            "missing_warnings": missing_warnings,
        }


# ==================== 单例 ====================

_review_agent: Optional[ReviewAgent] = None


def get_review_agent() -> ReviewAgent:
    """获取输出审核Agent单例"""
    global _review_agent
    if _review_agent is None:
        _review_agent = ReviewAgent()
    return _review_agent
