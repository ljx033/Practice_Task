---
name: pr-description-writer
description: Use after tests pass, to summarise the change as a short pull-request description. Triggers on summary, describe, done, PR, report, write-up.
---

# PR Description Writer

When the fix is verified (`run_tests` → `"passed": true`), call the `done` tool
with a summary that follows this shape:

- **What**: one sentence naming the file and function changed.
- **Why**: the bug the issue described (reversed operator, wrong variable, ...).
- **How verified**: `python -m pytest -q` → all tests passed.

Keep it under 5 lines. Do not invent tests you did not run.