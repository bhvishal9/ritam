from pydantic import BaseModel, ConfigDict


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class Cost(BaseModel):
    # A Cost that exists is always complete: all three are required, and the
    # value is frozen so it can't be half-mutated after construction. "Couldn't
    # price it" is represented as `Cost | None` at the boundary, not as a Cost
    # with None fields — that keeps "unpriced" distinct from a genuine $0.
    model_config = ConfigDict(frozen=True)

    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


class ModelPricing(BaseModel):
    input_tokens_price_1m: float
    output_tokens_price_1m: float
