"""Fixed 128K block layout and static query-anchor distance convention."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class BlockLayout:
    """Partition the pre-query cache into sink, learnable and recent blocks.

    The default layout has 1022 historical blocks. Query token zero begins at
    context_tokens and is the immutable distance anchor for the complete
    query/answer suffix.
    """

    total_tokens: int = 131_072
    context_tokens: int = 130_816
    block_size: int = 128
    sink_blocks: int = 1
    recent_blocks: int = 8
    needle_margin_tokens: int = 16

    def __post_init__(self) -> None:
        if self.total_tokens <= 0 or self.context_tokens <= 0:
            raise ValueError("token budgets must be positive")
        if self.context_tokens >= self.total_tokens:
            raise ValueError("context_tokens must leave a non-empty suffix budget")
        if self.block_size <= 0 or self.context_tokens % self.block_size:
            raise ValueError("context_tokens must be divisible by block_size")
        if self.sink_blocks < 1 or self.recent_blocks < 1:
            raise ValueError("sink_blocks and recent_blocks must be positive")
        if self.sink_blocks + self.recent_blocks >= self.num_context_blocks:
            raise ValueError("fixed blocks leave no learnable blocks")
        if self.needle_margin_tokens < 0:
            raise ValueError("needle_margin_tokens cannot be negative")
        if 2 * self.needle_margin_tokens >= self.block_size:
            raise ValueError("needle margins leave no usable block interior")

    @property
    def suffix_capacity_tokens(self) -> int:
        return self.total_tokens - self.context_tokens

    @property
    def num_context_blocks(self) -> int:
        return self.context_tokens // self.block_size

    @property
    def query_anchor_block(self) -> int:
        return self.num_context_blocks

    @property
    def learnable_start_block(self) -> int:
        return self.sink_blocks

    @property
    def learnable_stop_block(self) -> int:
        """Exclusive absolute-block boundary."""
        return self.num_context_blocks - self.recent_blocks

    @property
    def learnable_blocks(self) -> tuple[int, ...]:
        return tuple(range(self.learnable_start_block, self.learnable_stop_block))

    @property
    def num_learnable_blocks(self) -> int:
        return self.learnable_stop_block - self.learnable_start_block

    @property
    def recent_blocks_absolute(self) -> tuple[int, ...]:
        return tuple(range(self.learnable_stop_block, self.num_context_blocks))

    @property
    def usable_needle_tokens(self) -> int:
        return self.block_size - 2 * self.needle_margin_tokens

    @property
    def min_learnable_distance(self) -> int:
        return self.distance_block_id(self.learnable_stop_block - 1)

    @property
    def max_learnable_distance(self) -> int:
        return self.distance_block_id(self.learnable_start_block)

    def block_span(self, absolute_block_id: int) -> tuple[int, int]:
        if not 0 <= absolute_block_id < self.num_context_blocks:
            raise ValueError(f"invalid context block {absolute_block_id}")
        start = absolute_block_id * self.block_size
        return start, start + self.block_size

    def distance_block_id(self, absolute_block_id: int) -> int:
        if not 0 <= absolute_block_id < self.num_context_blocks:
            raise ValueError(f"invalid context block {absolute_block_id}")
        return self.query_anchor_block - absolute_block_id

    def gate_index(self, distance_block_id: int) -> int:
        if not self.min_learnable_distance <= distance_block_id <= self.max_learnable_distance:
            raise ValueError(
                f"distance {distance_block_id} is not learnable; expected "
                f"[{self.min_learnable_distance}, {self.max_learnable_distance}]"
            )
        return distance_block_id - self.min_learnable_distance

    def retention_mask_template(self) -> tuple[int | None, ...]:
        """Return fixed ones and None placeholders for learned decisions."""
        return (
            (1,) * self.sink_blocks
            + (None,) * self.num_learnable_blocks
            + (1,) * self.recent_blocks
        )

    def trainability_mask(self) -> tuple[int, ...]:
        return (
            (0,) * self.sink_blocks
            + (1,) * self.num_learnable_blocks
            + (0,) * self.recent_blocks
        )

    def as_dict(self) -> dict[str, int]:
        values = asdict(self)
        values.update(
            {
                "suffix_capacity_tokens": self.suffix_capacity_tokens,
                "num_context_blocks": self.num_context_blocks,
                "query_anchor_block": self.query_anchor_block,
                "learnable_start_block": self.learnable_start_block,
                "learnable_stop_block": self.learnable_stop_block,
                "num_learnable_blocks": self.num_learnable_blocks,
                "min_learnable_distance": self.min_learnable_distance,
                "max_learnable_distance": self.max_learnable_distance,
            }
        )
        return values
