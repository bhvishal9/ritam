import logging

from ritam.cost.types import Cost, ModelPricing, TokenUsage

MODELS: dict[str, ModelPricing] = {
    "gemini-3.1-flash-lite": ModelPricing(
        provider="gemini",
        input_tokens_price_1m=0.25,
        output_tokens_price_1m=1.50,
    ),
    "gemini-3-flash-preview": ModelPricing(
        provider="gemini",
        input_tokens_price_1m=0.50,
        output_tokens_price_1m=3.00,
    ),
}

logger = logging.getLogger(__name__)


def calculate_cost(model_name: str, token_usage: TokenUsage) -> Cost | None:
    """Compute the USD cost of a generation, or None if it can't be priced.

    Returns None when the model has no pricing entry or the provider didn't
    report token counts. None means "unpriced" and must stay distinct from a
    genuine zero cost so callers don't silently undercount spend.
    """
    model_pricing = MODELS.get(model_name)
    if model_pricing is None:
        logger.warning("model_not_priced", extra={"fields": {"model": model_name}})
        return None
    if token_usage.input_tokens is None or token_usage.output_tokens is None:
        logger.warning("token_usage_missing", extra={"fields": {"model": model_name}})
        return None

    input_cost = (
        token_usage.input_tokens / 1_000_000 * model_pricing.input_tokens_price_1m
    )
    output_cost = (
        token_usage.output_tokens / 1_000_000 * model_pricing.output_tokens_price_1m
    )
    return Cost(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )
