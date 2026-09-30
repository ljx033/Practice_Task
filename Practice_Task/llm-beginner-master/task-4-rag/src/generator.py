"""Qwen2.5-7B-Instruct 生成（OpenAI 兼容协议，默认对接 Ollama）。

支持的环境变量：
    RAG_LLM_BASE_URL   默认 http://localhost:11434/v1
    RAG_LLM_API_KEY    默认 "ollama"
    RAG_LLM_MODEL      默认 "qwen2.5:7b-instruct"
"""
from __future__ import annotations

import os
from typing import List

DEFAULT_BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_API_KEY = os.environ.get("RAG_LLM_API_KEY", "ollama")
DEFAULT_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen2.5:7b-instruct")

SYSTEM_PROMPT = (
    "你是一个严谨的中文问答助手。必须严格依据用户提供的【资料】回答问题，"
    "只能使用资料中明确出现的信息。"
    "如果资料不足以回答，请直接回答「根据提供的资料无法回答该问题」，"
    "禁止凭预训练知识补充或编造事实。回答要简洁、分点。"
)


def build_prompt(query: str, contexts: List[str], max_chars: int = 6000) -> str:
    """拼接上下文，做长度截断。"""
    picked: List[str] = []
    used = 0
    for i, c in enumerate(contexts, 1):
        block = f"[{i}] {c.strip()}"
        if used + len(block) > max_chars and picked:
            break
        picked.append(block)
        used += len(block)
    ctx = "\n\n".join(picked) if picked else "（无可用资料）"
    return f"【资料】\n{ctx}\n\n【问题】\n{query}\n\n请依据上面的资料作答。"


class Generator:
    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key: str = DEFAULT_API_KEY,
                 temperature: float = 0.1,
                 max_tokens: int = 512,
                 timeout: float = 120.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, contexts: List[str]) -> str:
        prompt = build_prompt(query, contexts)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()