from contextvars import copy_context

from ritam.observability.context import (
    bind_request_fields,
    generate_ms_context_var,
    request_id_context_var,
    request_scope,
    stage_context_var,
    total_cost_usd_context_var,
)
from ritam.observability.logging import _snapshot_contextvars


class TestRequestScopeIsolation:
    def test_unwritten_fields_do_not_leak_between_scopes(self) -> None:
        """An example that never generates must not inherit the previous cost.

        request_scope only sets request_id/dataset/top_k, so anything the
        pipeline skipped — generate_ms, cost, tokens — is cleared only because
        the scope binds a fresh store. Without that, an abstaining eval example
        reports the spend of whichever example ran before it.
        """
        with request_scope(request_id="first", dataset="ducks", top_k=3):
            generate_ms_context_var.set(1234.5)
            total_cost_usd_context_var.set(0.00042)

        with request_scope(request_id="second", dataset="ducks", top_k=3):
            assert generate_ms_context_var.get() is None
            assert total_cost_usd_context_var.get() is None

    def test_reading_a_field_does_not_bind_a_store(self) -> None:
        """Reads must not mutate the context.

        A read that binds leaks a store upward into the enclosing context. Since
        copy_context() copies the *binding*, every later copy would then share
        that one dict — so a single log line emitted before a loop would join
        every iteration of that loop into one store.
        """
        # Mimics a log line firing before the loop: the formatter reads every field.
        _snapshot_contextvars()

        seen: list[float | None] = []

        def example(cost: float | None) -> None:
            with request_scope(request_id="eval", dataset="ducks", top_k=3):
                if cost is not None:
                    total_cost_usd_context_var.set(cost)
                seen.append(total_cost_usd_context_var.get())

        copy_context().run(example, 0.00042)
        copy_context().run(example, None)

        assert seen == [0.00042, None]

    def test_scopes_restore_the_outer_store_on_exit(self) -> None:
        with bind_request_fields():
            request_id_context_var.set("outer")

            with request_scope(request_id="inner"):
                assert request_id_context_var.get() == "inner"

            assert request_id_context_var.get() == "outer"


class TestFieldSemantics:
    def test_get_returns_default_when_unset(self) -> None:
        with bind_request_fields():
            assert generate_ms_context_var.get() is None
            assert request_id_context_var.get() == "not-set"

    def test_reset_restores_previous_value(self) -> None:
        with bind_request_fields():
            token = stage_context_var.set("embed")
            assert stage_context_var.get() == "embed"
            stage_context_var.reset(token)
            assert stage_context_var.get() is None

    def test_snapshot_omits_unset_fields(self) -> None:
        with bind_request_fields():
            generate_ms_context_var.set(12.5)
            snapshot = _snapshot_contextvars()

        assert snapshot["generate_ms"] == 12.5
        assert "total_cost_usd" not in snapshot
