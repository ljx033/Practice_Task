---
name: code-review
description: Use when you must inspect a source file for a logic bug, or review a diff before committing. Triggers on words like bug, wrong, incorrect, review, inspect, verify, implementation.
---

# Code Review

A short, opinionated workflow for reading code critically.

## Procedure

1. Read the file top-to-bottom. For every function, note its name and docstring.
2. Check whether the **implementation matches its name and docstring**.
   - `add(a, b)` must return `a + b`, not `a - b`.
   - `factorial(n)` must raise on `n < 0`, loop from 2..n, return the product.
3. Scan for the classic small-bug patterns:
   - Reversed operator (`+` vs `-`, `<` vs `>`, `==` vs `!=`).
   - Off-by-one in `range(...)`.
   - Wrong variable shadowing an argument.
4. If a diff exists, run `git diff` and read only the changed hunks.
5. **Do not edit test files.** The tests express the intended behaviour.

## Output

A bullet list of concrete findings with `file:line` references and a one-line fix
per finding.