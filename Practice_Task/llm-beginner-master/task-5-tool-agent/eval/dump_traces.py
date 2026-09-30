"""批量跑指定题目，把完整 ReAct trace 导出成 markdown。

用法：
    python eval/dump_traces.py                # 默认跑有代表性的几题
    python eval/dump_traces.py 1 5 7 9        # 指定题号
    python eval/dump_traces.py --inject wiki  # 附带一条错误恢复 trace
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ReActAgent  # noqa: E402

OUT_DIR = ROOT / "eval" / "traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def format_trace(trace: dict) -> str:
    lines = [f"## 任务：{trace['task']}\n"]
    for step in trace.get("steps", []):
        lines.append(f"### Step {step.get('step', '?')}")

        # 模型原始输出：包含 Thought / Action / Action Input
        raw = step.get("raw")
        if raw:
            lines.append("```")
            lines.append(raw.strip())
            lines.append("```")

        # 兜底：如果 raw 缺失或结构化，显式补上 action/input
        if step.get("tool") and not raw:
            lines.append(f"Action: {step['tool']}")
            lines.append(
                f"Action Input: {json.dumps(step.get('tool_input', {}), ensure_ascii=False)}"
            )

        # 观察结果（工具返回值）
        if step.get("observation") is not None:
            obs = str(step["observation"])
            if len(obs) > 1200:
                obs = obs[:1200] + " …（截断）"
            lines.append(f"**Observation:** {obs}")

        # 错误 / 解析失败（错误恢复题会用到）
        if step.get("error"):
            lines.append(f"**[step error]** {step['error']}")
        if step.get("parse_error"):
            lines.append(f"**[parse error]** {step['parse_error']}")
        if step.get("final_answer"):
            lines.append(f"**Final Answer:** {step['final_answer']}")
        lines.append("")

    lines.append(f"**最终答案**: {trace.get('final_answer', '')}")
    lines.append(f"**self-reported success**: {trace.get('success')}")
    lines.append("")
    return "\n".join(lines)


def run_one(agent, task: dict) -> str:
    trace = agent.run(task["task"])
    return format_trace(trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int,
                    help="题号，如 1 5 7；不填则跑默认的几道")
    ap.add_argument("--inject", default=None,
                    help="注入错误的工具名（如 wiki），额外生成一条错误恢复 trace")
    args = ap.parse_args()

    tasks = json.loads((ROOT / "data" / "tasks.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in tasks}

    # 默认挑有代表性的：简单计算、两步多工具、内容检索、wiki+计算、读文件
    task_ids = args.ids or [1, 5, 7, 9, 10]

    agent = ReActAgent(verbose=False, max_steps=8, temperature=0.0)

    for tid in task_ids:
        t = by_id.get(tid)
        if t is None:
            print(f"跳过不存在的题号 {tid}")
            continue
        print(f"=== 跑第 {tid} 题：{t['task'][:40]}… ===")
        text = run_one(agent, t)
        path = OUT_DIR / f"trace_{tid:02d}.md"
        path.write_text(text, encoding="utf-8")
        print(f"  → 已写入 {path}")

    # 错误恢复 trace（S4）
    if args.inject:
        print(f"\n=== 附加：inject_error={args.inject} 的错误恢复 trace ===")
        agent_err = ReActAgent(
            verbose=False, max_steps=8, temperature=0.0,
            inject_error={"tool": args.inject, "times": 1},
        )
        # 挑一道会用到该工具的题
        target = next((t for t in tasks
                       if args.inject in t.get("expected_tools", [])), tasks[0])
        text = run_one(agent_err, target)
        path = OUT_DIR / f"trace_error_recovery_{args.inject}.md"
        path.write_text(text, encoding="utf-8")
        print(f"  → 已写入 {path}")

    print(f"\n全部完成，查看 {OUT_DIR}")


if __name__ == "__main__":
    main()