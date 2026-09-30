"""跑一次 CodingAgent，把 trace / patch 导出成提交材料。"""
import json
import sys
from pathlib import Path

from src.agent import CodingAgent

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "data" / "toy-repo"

issue = (REPO / "ISSUE.md").read_text(encoding="utf-8")
trace = CodingAgent().run(repo_path=str(REPO), issue=issue)

out_dir = ROOT / "eval" / "artifacts"
out_dir.mkdir(parents=True, exist_ok=True)

# 1) 完整 trace（steps / patch / tests_passed 全在）
(out_dir / "toy-repo-trace.json").write_text(
    json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

# 2) 单独的 patch 文件（unified diff，可直接 git apply）
patch = trace.get("patch", "")
(out_dir / "toy-repo.patch").write_text(patch, encoding="utf-8")

# 3) 控制台摘要
print("tests_passed:", trace.get("tests_passed"))
print("steps:", len(trace.get("steps", [])))
print("skills_loaded:", trace.get("skills_loaded"))
print("fallback_used:", trace.get("fallback_used", False))
print("patch bytes:", len(patch))
print("saved:", out_dir / "toy-repo-trace.json")
print("saved:", out_dir / "toy-repo.patch")
print("\n--- PATCH ---")
print(patch or "(empty)")