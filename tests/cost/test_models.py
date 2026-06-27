import pytest

from ritam.cost.models import calculate_cost
from ritam.cost.types import TokenUsage


class TestCalculateCost:
    def test_prices_known_model_with_token_counts(self) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        cost = calculate_cost("gemini-3.1-flash-lite", usage)

        # input: 1000/1e6 * $0.25 = $0.00025; output: 500/1e6 * $1.50 = $0.00075
        assert cost is not None
        assert cost.input_cost_usd == pytest.approx(0.00025)
        assert cost.output_cost_usd == pytest.approx(0.00075)
        assert cost.total_cost_usd == pytest.approx(0.001)

    def test_unknown_model_is_unpriced(self) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        assert calculate_cost("model-not-in-table", usage) is None

    def test_missing_token_counts_is_unpriced(self) -> None:
        # Provider reported no usage_metadata, or only a partial count.
        assert calculate_cost("gemini-3.1-flash-lite", TokenUsage()) is None
        assert (
            calculate_cost("gemini-3.1-flash-lite", TokenUsage(input_tokens=10)) is None
        )
