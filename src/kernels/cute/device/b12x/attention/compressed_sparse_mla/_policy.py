"""Typed component policy for compressed sparse MLA planning."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.policy import (
    COMPRESSED_SPARSE_MLA_ATTENTION,
    ComponentPolicy,
    DeviceIdentity,
    FrozenMapping,
)


@dataclass(frozen=True, kw_only=True)
class SparseMlaQuery:
    layout: str
    mode: str
    q_dtype: str
    kv_dtype: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    swa_width: int
    swa_page_size: int
    indexed_width: int
    indexed_page_size: int
    query_rows: int
    cache_format: str = "deepseek_v4"

    def profile_fields(self) -> dict[str, object]:
        return {
            "layout": self.layout,
            "cache_format": self.cache_format,
            "mode": self.mode,
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "num_q_heads": self.num_q_heads,
            "qk_head_dim": self.qk_head_dim,
            "v_head_dim": self.v_head_dim,
            "swa_width": self.swa_width,
            "swa_page_size": self.swa_page_size,
            "indexed_width": self.indexed_width,
            "indexed_page_size": self.indexed_page_size,
            "query_rows": self.query_rows,
        }


@dataclass(frozen=True, kw_only=True)
class SparseMlaConfig:
    max_chunks_per_row: int
    v41_compute_mode: str = "fp8"
    v41_heads_per_block: int = 16

    @classmethod
    def from_profile(cls, payload: FrozenMapping) -> "SparseMlaConfig":
        required = {
            "max_chunks_per_row",
            "v41_compute_mode",
            "v41_heads_per_block",
        }
        if set(payload) != required:
            raise ValueError(
                "sparse MLA profiles require exactly "
                "max_chunks_per_row, v41_compute_mode, and v41_heads_per_block"
            )
        max_chunks_per_row = payload["max_chunks_per_row"]
        v41_compute_mode = payload["v41_compute_mode"]
        v41_heads_per_block = payload["v41_heads_per_block"]
        if not isinstance(max_chunks_per_row, int) or isinstance(
            max_chunks_per_row, bool
        ):
            raise TypeError("sparse MLA max_chunks_per_row must be an integer")
        if not isinstance(v41_compute_mode, str):
            raise TypeError("sparse MLA v41_compute_mode must be a string")
        if not isinstance(v41_heads_per_block, int) or isinstance(
            v41_heads_per_block, bool
        ):
            raise TypeError("sparse MLA v41_heads_per_block must be an integer")
        return cls(
            max_chunks_per_row=max_chunks_per_row,
            v41_compute_mode=v41_compute_mode,
            v41_heads_per_block=v41_heads_per_block,
        )


def _heuristic(
    query: SparseMlaQuery,
    device: DeviceIdentity | None,
) -> SparseMlaConfig:
    capability = None if device is None else device.compute_capability
    uses_single_pass = query.mode != "decode" or (
        query.cache_format == "deepseek_v4"
        and capability == (12, 1)
        and query.query_rows >= 16
        and query.num_q_heads == 32
        and query.swa_page_size == 64
        and (query.indexed_width == 0 or query.indexed_page_size == 64)
    )
    return SparseMlaConfig(
        max_chunks_per_row=1 if uses_single_pass else 64,
        v41_compute_mode="fp8",
        v41_heads_per_block=16 if query.num_q_heads % 16 == 0 else 8,
    )


def _validate(
    _query: SparseMlaQuery,
    config: SparseMlaConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, SparseMlaConfig):
        raise TypeError("sparse MLA config must be SparseMlaConfig")
    if (
        not isinstance(config.max_chunks_per_row, int)
        or isinstance(config.max_chunks_per_row, bool)
    ):
        raise TypeError("sparse MLA max_chunks_per_row must be an integer")
    if config.max_chunks_per_row <= 0:
        raise ValueError("sparse MLA max_chunks_per_row must be positive")
    if not isinstance(config.v41_compute_mode, str):
        raise TypeError("sparse MLA v41_compute_mode must be a string")
    if config.v41_compute_mode not in {"fp8", "bf16"}:
        raise ValueError("sparse MLA v41_compute_mode must be 'fp8' or 'bf16'")
    if (
        not isinstance(config.v41_heads_per_block, int)
        or isinstance(config.v41_heads_per_block, bool)
    ):
        raise TypeError("sparse MLA v41_heads_per_block must be an integer")
    if config.v41_heads_per_block not in {8, 16}:
        raise ValueError("sparse MLA v41_heads_per_block must be 8 or 16")


COMPRESSED_SPARSE_MLA_POLICY = ComponentPolicy(
    component_id=COMPRESSED_SPARSE_MLA_ATTENTION,
    query_schema_version=2,
    config_schema_version=2,
    query_fields=frozenset(
        {
            "layout",
            "cache_format",
            "mode",
            "q_dtype",
            "kv_dtype",
            "num_q_heads",
            "qk_head_dim",
            "v_head_dim",
            "swa_width",
            "swa_page_size",
            "indexed_width",
            "indexed_page_size",
            "query_rows",
        }
    ),
    config_fields=frozenset(
        {"max_chunks_per_row", "v41_compute_mode", "v41_heads_per_block"}
    ),
    encode_query=SparseMlaQuery.profile_fields,
    decode_profile=SparseMlaConfig.from_profile,
    heuristic=_heuristic,
    validate_config=_validate,
)
SPARSE_MLA_POLICY = COMPRESSED_SPARSE_MLA_POLICY


__all__ = [
    "COMPRESSED_SPARSE_MLA_POLICY",
    "SPARSE_MLA_POLICY",
    "SparseMlaConfig",
    "SparseMlaQuery",
]
