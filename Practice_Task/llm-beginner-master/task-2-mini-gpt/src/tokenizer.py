"""字节级 BPE 分词器。

设计要点：
- 字节级：text -> utf-8 bytes -> ids；decode 先拼字节再 decode utf-8，跨多字节边界安全。
- 训练：在整段训练文本的字节序列上迭代合并「当前频率最高的相邻 token 对」。
- 编码：用「最小合并优先级优先」的贪心策略，借助双向链表 + 小顶堆实现 O(n log n)。
- 词表：0-255 为基础字节，256+ 为合并产生的新 token。
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Dict, List, Tuple


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: Dict[Tuple[int, int], int] = {}
        self._vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ---------------- 基本属性 ----------------
    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    # ---------------- 训练 ----------------
    def train(self, text: str, vocab_size: int = 1024, verbose: bool = False) -> None:
        """在给定文本上训练 BPE。"""
        if vocab_size <= 256:
            raise ValueError("vocab_size 必须大于 256")

        ids: List[int] = list(text.encode("utf-8"))
        merges: Dict[Tuple[int, int], int] = {}
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for new_id in range(256, vocab_size):
            if len(ids) < 2:
                break
            # 统计相邻对频次
            cnt: Dict[Tuple[int, int], int] = {}
            for i in range(len(ids) - 1):
                p = (ids[i], ids[i + 1])
                cnt[p] = cnt.get(p, 0) + 1
            if not cnt:
                break
            # 最高频，平局按 (a,b) 升序，保证确定性
            best_pair, best_count = max(
                cnt.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1])
            )
            if best_count < 2:
                break
            a, b = best_pair
            merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]

            # 应用合并
            new_ids: List[int] = []
            i, n = 0, len(ids)
            while i < n:
                if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

            if verbose and (new_id - 255) % 200 == 0:
                print(f"  [BPE] merge {new_id - 255}, seq_len={len(ids)}")

        self.merges = merges
        self._vocab = vocab

    # ---------------- 编码 ----------------
    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        ids = list(text.encode("utf-8"))
        return self._apply_bpe(ids)

    def _apply_bpe(self, ids: List[int]) -> List[int]:
        """用双向链表 + 小顶堆按合并优先级贪心应用 merges。"""
        if not self.merges or len(ids) < 2:
            return list(ids)

        n = len(ids)
        ids = list(ids)
        prev = list(range(-1, n - 1))   # prev[i] = i-1
        nxt = list(range(1, n + 1))     # nxt[i] = i+1
        nxt[-1] = -1
        alive = [True] * n

        heap: List[Tuple[int, int, int, int]] = []

        def push(i: int) -> None:
            if i < 0 or i >= n or not alive[i]:
                return
            j = nxt[i]
            if j == -1 or not alive[j]:
                return
            pri = self.merges.get((ids[i], ids[j]))
            if pri is not None:
                heapq.heappush(heap, (pri, i, ids[i], ids[j]))

        for i in range(n - 1):
            push(i)

        while heap:
            _pri, i, a, b = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if ids[i] != a or ids[j] != b:
                continue
            new_id = self.merges[(a, b)]
            ids[i] = new_id
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] != -1:
                prev[nxt[j]] = i
            if prev[i] != -1:
                push(prev[i])
            push(i)

        return [ids[i] for i in range(n) if alive[i]]

    # ---------------- 解码 ----------------
    def decode(self, ids) -> str:
        if ids is None:
            return ""
        parts: List[bytes] = []
        for i in ids:
            b = self._vocab.get(int(i))
            if b is not None:
                parts.append(b)
        # errors="replace" 保证未登录字节序列也不抛异常
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---------------- 存取 ----------------
    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        merges_list = sorted(self.merges.items(), key=lambda kv: kv[1])
        data = {
            "format": "mini-gpt-bpe-v1",
            "vocab_size": self.vocab_size,
            # 每条为 [a, b, new_id]，按 new_id 升序（即合并优先级顺序）
            "merges": [[int(a), int(b), int(i)] for (a, b), i in merges_list],
        }
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        tok = cls()
        merges: Dict[Tuple[int, int], int] = {}
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for a, b, new_id in data["merges"]:
            a, b, new_id = int(a), int(b), int(new_id)
            merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]
        tok.merges = merges
        tok._vocab = vocab
        return tok