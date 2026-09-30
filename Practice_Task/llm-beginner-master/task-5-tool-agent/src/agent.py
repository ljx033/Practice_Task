"""手写 ReAct 循环：Thought / Action / Action Input / Observation / Final Answer。

- 工具路由：解析模型输出，分派到 4 个本地工具
- 错误恢复：工具抛异常 → 塞回 Observation，让模型自我纠错（M3）
- 终止条件：Final Answer 或步数上限
"""
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.tools import calculator, python_sandbox, file_search, wiki

_TOOLS = {
    "calculator": calculator,
    "python_sandbox": python_sandbox,
    "file_search": file_search,
    "wiki": wiki,
}

_REQUIRED_KEYS = {
    "calculator": ["expression"],
    "python_sandbox": ["code"],
    "file_search": ["pattern", "dir"],
    "wiki": ["query"],
}

_DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
_DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
_DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5:7b-instruct")

SYSTEM_PROMPT = """你是一个会使用工具解决任务的智能体。严格按 ReAct 格式逐步推理和行动。

# 可用工具
1. calculator      —— 数学表达式求值
   参数：{"expression": "<数学表达式>"}
2. python_sandbox  —— 受限 Python 执行（捕获 print 输出）
   参数：{"code": "<Python 代码>"}
3. file_search     —— 本地文件按名/内容检索（返回路径和内容片段，也可直接读文件）
   参数：{"pattern": "<文件名或内容关键字，支持 *>", "dir": "<搜索根目录>"}
4. wiki            —— 维基百科条目查询（中英文都支持）
   参数：{"query": "<词条名>"}

# 输出格式（严格遵守）
Thought: 分析当前情况，决定下一步
Action: 工具名（上面 4 个之一）
Action Input: 该工具参数的 JSON 对象（必须合法 JSON，字符串用双引号）

系统返回 Observation 后，你再继续下一轮 Thought。
当你已能回答用户时，直接输出：
Thought: 我已经得到最终答案
Final Answer: <最终答案，必须直接包含题目要求的关键数字 / 关键词>

# 规则
- 每次只输出一个 Action，不要自己编造 Observation。
- 多步任务分步执行：先调一个工具，看 Observation 再决定下一步。
- 涉及精度时用 round(expr, 6) 或让 Python 输出足够位数。
- 若工具返回错误，请你分析错误并调整 Action Input 重试。
- Final Answer 里必须包含题目要求的关键信息（数字、文件名、年份、人名等）。

# 示例
用户：计算 (12 + 8) * 5。
Thought: 用 calculator。
Action: calculator
Action Input: {"expression": "(12 + 8) * 5"}
Observation: (12 + 8) * 5 = 100
Thought: 已经得到答案
Final Answer: 结果是 100。

用户：用 Python 计算 1 到 100 的和。
Thought: 用 python_sandbox。
Action: python_sandbox
Action Input: {"code": "print(sum(range(1, 101)))"}
Observation: 5050
Thought: 已经得到答案
Final Answer: 1 到 100 的和是 5050。

用户：在 data/agent-fixtures 下查找包含 TODO 的文件。
Thought: 用 file_search 做内容搜索。
Action: file_search
Action Input: {"pattern": "TODO", "dir": "data/agent-fixtures"}
Observation: 在 ... 下找到 1 个匹配 'TODO' 的文件：.../todo_note.md
Thought: 已经得到答案
Final Answer: 包含 TODO 的文件是 data/agent-fixtures/todo_note.md。
"""


# ------------------------- 解析工具函数 -------------------------

def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_loose(raw: str, tool_name: Optional[str] = None) -> Dict[str, Any]:
    raw = _strip_fence(raw)
    if not raw:
        return {}
    # ① 直接 JSON
    try:
        v = json.loads(raw)
        if isinstance(v, dict):
            return v
        if isinstance(v, str) and tool_name:
            return {_REQUIRED_KEYS.get(tool_name, ["input"])[0]: v}
        if isinstance(v, (int, float)) and tool_name:
            return {_REQUIRED_KEYS.get(tool_name, ["input"])[0]: str(v)}
    except Exception:
        pass
    # ② 提取 {...}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    # ③ key="value" / key='value'
    kv = {}
    for pat in (r'([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*"([^"]*)"',
                r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*'([^']*)'"):
        for k, v in re.findall(pat, raw):
            kv[k] = v
        if kv:
            break
    if kv:
        if tool_name:
            need = _REQUIRED_KEYS.get(tool_name, [])
            if need and need[0] not in kv:
                return {need[0]: raw}
        return kv
    # ④ 兜底：整段当第一个必需参数
    if tool_name:
        need = _REQUIRED_KEYS.get(tool_name, ["input"])
        val = raw.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        return {need[0]: val}
    return {}


def _parse_action(text: str):
    action = None
    m = re.search(r"(?:^|\n)\s*Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_\-]*)", text)
    if m:
        action = m.group(1).strip()
    m2 = re.search(
        r"(?:^|\n)\s*Action\s*Input\s*[:：]\s*(.*?)"
        r"(?=\n\s*(?:Thought|Action|Observation|Final\s*Answer)\s*[:：]|\Z)",
        text, re.DOTALL,
    )
    raw_input = m2.group(1).strip() if m2 else ""
    parsed = _parse_json_loose(raw_input, action) if raw_input else {}
    # 归一化工具名
    if action:
        low = action.lower().replace("-", "_")
        for name in _TOOLS:
            if low == name:
                action = name
                break
    return action, parsed, raw_input


def _extract_final(text: str) -> Optional[str]:
    m = re.search(r"Final\s*Answer\s*[:：]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        if ans:
            return ans
    return None


# ------------------------- Agent 主体 -------------------------

class ReActAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_steps: int = 8,
        temperature: float = 0.0,
        request_timeout: float = 120.0,
        verbose: bool = False,
        inject_error: Optional[Dict[str, Any]] = None,
    ):
        self.model = model or _DEFAULT_MODEL
        self.base_url = base_url or _DEFAULT_BASE_URL
        self.api_key = api_key or _DEFAULT_API_KEY
        self.max_steps = max_steps
        self.temperature = temperature
        self.verbose = verbose
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                             timeout=request_timeout)
        self.tools = dict(_TOOLS)
        # S4 错误注入：{"tool": "wiki", "times": 1}
        self.inject_error = inject_error
        self._inject_remaining = int((inject_error or {}).get("times", 0) or 0)

    # -- LLM --
    def _call_llm(self, messages) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stop=["\nObservation:", "Observation:"],
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    # -- 工具执行（含错误捕获 → M3） --
    def _run_tool(self, action: str, args: dict) -> str:
        if action not in self.tools:
            return f"错误：未知工具 '{action}'。可用工具：{list(self.tools.keys())}"
        # 错误注入钩子
        if (self.inject_error and self._inject_remaining > 0
                and self.inject_error.get("tool") == action):
            self._inject_remaining -= 1
            return f"错误（注入测试）：{action} 临时不可用，请换一种方式或重试。"
        try:
            return str(self.tools[action].run(args))
        except KeyError as e:
            return (f"参数错误（KeyError: {e}）。工具 {action} 需要参数键 "
                    f"{_REQUIRED_KEYS.get(action)}，请修正 Action Input。")
        except Exception as e:
            return f"工具执行出错（{type(e).__name__}）：{e}"

    # -- 主循环 --
    def run(self, task: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"任务：{task}"},
        ]
        steps: List[Dict[str, Any]] = []
        parse_failures = 0
        final_answer = ""
        success = False

        for i in range(self.max_steps):
            try:
                text = self._call_llm(messages)
            except Exception as e:
                steps.append({"step": i + 1, "error": f"LLM 调用失败：{e}"})
                break
            if self.verbose:
                print(f"\n--- Step {i + 1} ---\n{text}\n")
            messages.append({"role": "assistant", "content": text})
            step: Dict[str, Any] = {"step": i + 1, "raw": text}

            # Final Answer 优先
            fa = _extract_final(text)
            if fa:
                step["final_answer"] = fa
                steps.append(step)
                final_answer = fa
                success = True
                break

            action, args, raw_input = _parse_action(text)
            step["raw_action_input"] = raw_input

            if not action:
                parse_failures += 1
                step["parse_error"] = "未检测到 Action"
                steps.append(step)
                if parse_failures >= 3:
                    break
                messages.append({
                    "role": "user",
                    "content": ("格式错误：没有检测到 `Action:` 行。请严格按格式输出 "
                                "`Thought: ...`、`Action: 工具名`、`Action Input: {...}`；"
                                "如果已得到答案则用 `Final Answer: ...` 结束。"),
                })
                continue

            if not args:
                parse_failures += 1
                step["parse_error"] = "Action Input 解析失败"
                step["action"] = action
                step["tool"] = action
                steps.append(step)
                messages.append({
                    "role": "user",
                    "content": (f"Action Input 解析失败。工具 {action} 需要参数键 "
                                f"{_REQUIRED_KEYS.get(action)}，请输出合法 JSON，"
                                f"例如 {{\"expression\": \"2+3\"}}。"),
                })
                continue

            obs = self._run_tool(action, args)
            step["action"] = action
            step["tool"] = action
            step["tool_input"] = args
            step["observation"] = obs
            steps.append(step)
            if self.verbose:
                print(f"Action: {action}\nAction Input: {args}\nObservation: {obs}\n")
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        # 兜底：未拿到 Final Answer 时强制收尾
        if not success:
            messages.append({
                "role": "user",
                "content": ("请基于以上所有 Observation 给出最终答案，"
                            "格式必须以 `Final Answer:` 开头。"),
            })
            try:
                text = self._call_llm(messages)
                fa = _extract_final(text) or text.strip()
                steps.append({"step": len(steps) + 1, "raw": text,
                              "final_answer": fa, "forced": True})
                final_answer = fa
                success = bool(fa)
            except Exception as e:
                steps.append({"step": len(steps) + 1,
                              "error": f"强制收尾失败：{e}"})

        return {
            "task": task,
            "steps": steps,
            "final_answer": final_answer,
            "success": bool(success),
        }


# 兼容 eval/run.py 的类型别名
AgentTrace = Dict[str, Any]