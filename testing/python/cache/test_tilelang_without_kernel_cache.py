"""Tests for ``tilelang.testing.without_kernel_cache``."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import env
from tilelang.env import CacheState
from tilelang.jit import JITImpl


@pytest.fixture(autouse=True)
def _restore_cache_state():
    enabled = CacheState.is_enabled()
    try:
        yield
    finally:
        if enabled:
            CacheState.enable()
        else:
            CacheState.disable()


def _set_cache_state(enabled: bool) -> None:
    if enabled:
        CacheState.enable()
    else:
        CacheState.disable()


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_without_kernel_cache_restores_the_previous_state(initially_enabled):
    _set_cache_state(initially_enabled)

    with tilelang.testing.without_kernel_cache():
        assert not CacheState.is_enabled()

    assert CacheState.is_enabled() is initially_enabled


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_without_kernel_cache_restores_the_previous_state_after_the_body_reenables(initially_enabled):
    _set_cache_state(initially_enabled)

    with tilelang.testing.without_kernel_cache():
        CacheState.enable()

    assert CacheState.is_enabled() is initially_enabled


def test_without_kernel_cache_bypasses_the_jit_wrapper_caches(monkeypatch):
    """A jit call inside the block compiles instead of reusing an earlier kernel."""
    # Pin both switches instead of inheriting them from the caller's environment.
    monkeypatch.setattr(vars(type(env))["TILELANG_DISABLE_CACHE"], "_forced_value", "0")
    _set_cache_state(True)
    compile_calls = []

    def fake_compile(self, *args, **kwargs):
        compile_calls.append(args)
        return object()

    monkeypatch.setattr(JITImpl, "compile", fake_compile)

    @tilelang.jit(out_idx=-1, target="c")
    def build(N):
        @T.prim_func
        def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
            with T.Kernel(1, threads=128):
                for i in T.Parallel(N):
                    C[i] = A[i] + B[i]

        return main

    first = build(128)
    assert build(128) is first
    assert len(compile_calls) == 1

    with tilelang.testing.without_kernel_cache():
        fresh = build(128)
        assert build(128) is fresh

    assert fresh is not first
    assert len(compile_calls) == 2

    assert build(128) is first
    assert build(N=128) is first
    assert len(compile_calls) == 2

    # A plain disable_cache() (Metal, env var) does not open a fresh scope and
    # must keep the wrapper caches working.
    env.disable_cache()
    assert build(128) is first
    assert len(compile_calls) == 2


if __name__ == "__main__":
    tilelang.testing.main()


def test_without_kernel_cache_releases_scoped_wrapper_entries(monkeypatch):
    """Scoped wrapper entries must not outlive the scope."""
    monkeypatch.setattr(vars(type(env))["TILELANG_DISABLE_CACHE"], "_forced_value", "0")
    _set_cache_state(True)

    def fake_compile(self, *args, **kwargs):
        return object()

    monkeypatch.setattr(JITImpl, "compile", fake_compile)

    @tilelang.jit(out_idx=-1, target="c")
    def build(N):
        @T.prim_func
        def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
            with T.Kernel(1, threads=128):
                for i in T.Parallel(N):
                    C[i] = A[i] + B[i]

        return main

    baseline = build(128)
    with tilelang.testing.without_kernel_cache():
        build(128)

    assert build(128) is baseline
    assert not any(isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], int)
                   for key in build._kernel_cache)
    assert not any(isinstance(key, tuple) and len(key) == 3 and isinstance(key[2], int) and key[2] != 0
                   for key in build._call_form_cache.entries)


def test_without_kernel_cache_scopes_can_exit_out_of_order():
    """Overlapping scopes keep the original state and clean up independently."""
    initially_enabled = CacheState.is_enabled()
    first = CacheState.begin_scope()
    second = CacheState.begin_scope()
    try:
        assert CacheState.is_enabled() is False
        CacheState.end_scope(first)
        assert CacheState.is_enabled() is False
    finally:
        CacheState.end_scope(second)
    assert CacheState.is_enabled() is initially_enabled


def test_without_kernel_cache_preserves_body_exception(monkeypatch):
    monkeypatch.setattr(vars(type(env))["TILELANG_DISABLE_CACHE"], "_forced_value", "0")
    _set_cache_state(True)
    with pytest.raises(RuntimeError, match="body failed"):
        with tilelang.testing.without_kernel_cache():
            raise RuntimeError("body failed")
    assert CacheState.is_enabled() is True


def test_without_kernel_cache_scopes_are_thread_local():
    """Concurrent scopes must not share a stack entry."""
    import threading

    initially_enabled = CacheState.is_enabled()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker():
        try:
            barrier.wait()
            with tilelang.testing.without_kernel_cache():
                assert CacheState.is_enabled() is False
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert CacheState.is_enabled() is initially_enabled
