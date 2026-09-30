---
name: test-runner
description: Use when running tests, diagnosing a pytest failure, or confirming a fix made the suite green. Triggers on test, pytest, failed, failing, assertion, green, pass.
---

# Test Runner

## Procedure

1. Always run `python -m pytest -q` from the repo root (use the `run_tests` tool).
2. Read the **first** failure only. Later failures are often cascades of the first.
3. From the assertion, extract `expected` vs `actual` and map it back to the
   exact source line. Example:
E assert -1 == 5

text

→ `add(2, 3)` returned `-1`; the implementation subtracts instead of adds.

4. Fix the **source**, never the test.
5. Re-run. Repeat until the tool reports `"passed": true`.

## Common traps

- The bug is frequently a single operator (`+` ↔ `-`, `<=` ↔ `<`).
- Do not use `--ignore=` or `-k` to skip failures.
- If pytest itself cannot import the module, the cause is usually a syntax
error you just introduced — re-read the file you wrote.