#!/usr/bin/env bash
# run_examples.sh - run all Python examples under ./examples

set -u
shopt -s nullglob

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Ensure local imports resolve (PyDynSys under src/)
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

EXIT=0
PASSED=0
FAILED=0

EXAMPLES=(examples/*.py)

if [ ${#EXAMPLES[@]} -eq 0 ]; then
  echo "No example .py files found under ./examples"
  exit 1
fi

for file in "${EXAMPLES[@]}"; do
  echo "==> Running $file"
  if python "$file"; then
    echo "OK: $file"
    PASSED=$((PASSED+1))
  else
    echo "FAIL: $file"
    FAILED=$((FAILED+1))
    EXIT=1
  fi
  echo
done

echo "Summary: ${PASSED} passed, ${FAILED} failed"
exit $EXIT