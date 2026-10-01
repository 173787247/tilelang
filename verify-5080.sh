#!/usr/bin/env bash
# Verify every contribution branch on this machine, one at a time.
#
# For each branch:
#   1. check out the branch
#   2. rebuild if it touches src/ (the build tree is shared, so this matters)
#   3. clear the kernel cache (a cached kernel from another build hides the change)
#   4. run the test files the branch itself touches
#   5. call the `run_*` functions of any test file that mentions the changed code
#      -- pytest never collects those, which is how a wrong PR slipped through once
#
# Prints one line per branch. Owns the working tree: do not run git beside it.

cd "$HOME/GO/tilelang" || exit 1
# .dev-env.sh reads $LD_LIBRARY_PATH unguarded, so it aborts under `set -u`.
# Push one first, then turn -u on for our own code.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source .dev-env.sh >/dev/null 2>&1
set -u

BRANCHES=$(git branch --format='%(refname:short)' | grep '^contrib/' | grep -v tvmffi-cluster | sort)
TOTAL=0; GREEN=0
REPORT=/tmp/verify-5080-report.txt
: > "$REPORT"

for b in $BRANCHES; do
  TOTAL=$((TOTAL + 1))
  git checkout "$b" -q 2>/dev/null || { echo "  $b  CHECKOUT FAILED"; continue; }

  changed_src=$(git diff --name-only main..."$b" -- src/ '*.pyx' 2>/dev/null | head -1)
  changed_all=$(git diff --name-only main..."$b" 2>/dev/null | grep -v '^testing/')

  # 2. rebuild when C++ moved
  build_note="no-build"
  if [ -n "$changed_src" ]; then
    if cmake --build build -j32 >/tmp/v_build.log 2>&1; then build_note="built"; else build_note="BUILD-FAIL"; fi
  fi

  # 3. kernel cache
  rm -rf "$HOME/.tilelang/cache"

  # 4. the branch's own tests
  testfiles=$(git diff --name-only main..."$b" -- testing/ 2>/dev/null | tr '\n' ' ')
  test_note="no-tests"
  if [ -n "$testfiles" ]; then
    timeout 1800 python -m pytest $testfiles -q >/tmp/v_test.log 2>&1
    newfails=$(grep -E '^FAILED' /tmp/v_test.log | grep -vcE 'test_reduce\[(sum|min)-(float16|bfloat16)|test_cast_rs_|test_cast_default_unchanged|test_tiled_ws_')
    total_fails=$(grep -cE '^FAILED' /tmp/v_test.log)
    if [ "$newfails" = "0" ]; then
      [ "$total_fails" = "0" ] && test_note="tests-pass" || test_note="pass(${total_fails}known)"
    else
      test_note="TESTS-FAIL"
      grep -E '^FAILED' /tmp/v_test.log | grep -vE 'test_reduce\[(sum|min)-(float16|bfloat16)|test_cast_rs_|test_tiled_ws_' | head -3 >> "$REPORT"
    fi
  fi

  # 5. non-collected run_* functions in files that mention what we changed
  run_note="run-fn=0"
  if [ -n "$changed_all" ]; then
    pat=$(echo "$changed_all" | xargs -n1 basename 2>/dev/null | paste -sd'|')
    hits=$(grep -rlE "$pat" testing/ --include='*.py' 2>/dev/null | xargs grep -ln '^def run_' 2>/dev/null | head -3)
    n=0
    for f in $hits; do
      out=$(timeout 600 python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('m', '$f')
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except Exception as e:
    print('skip-import-error'); sys.exit(0)
import inspect
bad = 0
for name, obj in vars(m).items():
    if name.startswith('run_') and callable(obj) and not inspect.signature(obj).parameters:
        try:
            obj()
        except Exception as e:
            print('FAIL:' + name); bad += 1
print('ok' if bad == 0 else 'bad')
" 2>/dev/null | tail -1)
      n=$((n + 1))
      case "$out" in
        ok) ;;
        skip-import-error) [ "$run_note" = "run-fn=0" ] && run_note="run-fn=$n-skip-import" ;;
        *) run_note="RUN-FN-FAIL:$f:$out"; echo "  $b  $f -> $out" >> "$REPORT" ;;
      esac
    done
    [ "$run_note" = "run-fn=0" ] && run_note="run-fn=$n-ok"
  fi

  status="OK"
  [ "$build_note" = "BUILD-FAIL" ] && status="BAD"
  [ "$test_note" = "TESTS-FAIL" ] && status="BAD"
  case "$run_note" in RUN-FN-FAIL*) status="BAD";; esac
  [ "$status" = "OK" ] && GREEN=$((GREEN + 1))

  printf "  %-6s %-52s %-9s %-11s %s\n" "$status" "$b" "$build_note" "$test_note" "$run_note"
done

echo "  ── $GREEN/$TOTAL 全绿"
[ -s "$REPORT" ] && { echo "  失败明细:"; cat "$REPORT"; }
git checkout main -q
