# Verification sweep — 23 pull requests against a single machine

This is the contributor's own check of the contributions referenced from
tile-ai/tilelang#3301, and the other reports it links. It is **not** independent
verification: the same author wrote the fixes and ran the sweep. What it does
establish is that each branch was exercised end to end, on real hardware, in a
state that could actually fail.

Everything below was produced by `verify-5080.sh` (in the same branch), one
branch at a time.

## Method

For each contribution branch:

1. check out the branch;
2. **rebuild** when the branch touches anything CMake compiles — `src/` **or**
   `*.pyx`. Getting this wrong produces false results in both directions: a
   stale `.so` makes a good branch look broken, and a stale kernel hides the
   change being tested;
3. **clear the kernel cache** (`~/.tilelang/cache`). A cached kernel compiled
   from a fixed tree will otherwise mask both a reproducer and a regression;
4. run the test files the branch itself touches;
5. call the `run_*` functions of any test file that mentions the changed code.
   pytest collects `test_*` only, and this repository has 190 `run_*` functions
   across 62 files. They are ordinary assertions that never run in CI, which is
   how a wrong change got past step 4 once already.

## Result

```
20 branches clean
 3 branches showing only failures that pre-exist on this machine (below)
 0 unexplained failures

run_* check: no failures on any branch
```

## The three branches that are not fully green

Both causes are environment limits, and both reproduce on an unmodified tree:

| Branch | Failing tests | Cause |
|---|---|---|
| `contrib/3301-batch-reduce-thread-range` | `test_reduce[sum-float16-64x128-f2f-t256-b4]`, `[sum-bfloat16-...]`, `[min-float16-128x128-f2f-t256-b8]` | pre-existing: `Check failed: batch <= N_total` |
| `contrib/3301-bitwise-reduce-source-dtype` | the same three | same |
| `contrib/3301-vectorized-cast-round` | `test_cast_rs_*` in `test_tilelang_cast_rounding.py` | pre-existing: this machine is `sm_120`, those need `sm_100a` |

```
tl_templates/cuda/cuda_fp4.h(313): static assertion failed with
  "Stochastic rounding f32-to-FP4 requires sm_100a or sm_103a"
```

## Environment

```
GPU          NVIDIA GeForce RTX 5080, compute capability 12.0 (sm_120)
Driver       591.86   |   CUDA toolkit 13.0.88
Host         32 cores, WSL2
```

## Two fixes to the harness, found by running it

- The first run reported a branch as failing because the harness rebuilt only on
  `src/`. `cython_wrapper.pyx` is compiled by CMake, so the stale module was
  used; with the right rebuild condition the same branch is 11 passed.
- `.dev-env.sh` reads `$LD_LIBRARY_PATH` unguarded, so under `set -u` sourcing
  it terminates the shell outright (`LD_LIBRARY_PATH: unbound variable`). A
  harness that fails silently is worse than one that fails loudly.
