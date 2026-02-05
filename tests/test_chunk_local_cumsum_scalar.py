# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import sys
from collections.abc import Callable
from typing import Any, Literal
import functools
import warnings
import torch
import triton
import triton.language as tl

from tests.register_ops import chunk_local_cumsum_scalar_kernel

DEVICE = torch.device("xpu")

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "seed": [123],
        "dtype": [torch.float16],
        "batch": [1],
        "token": [4],
        "head": [32],
        "chunk_size": [64],
        "is_varlen": [True],
        "reverse": [False],
        "head_first": [False],
        "output_dtype": [torch.float32],
    },
}

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    print(cu_seqlens, prepare_lens(cu_seqlens), cu_seqlens[1:], cu_seqlens[:-1])
    print(triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist())
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    print(indices, [indices.eq(0).cumsum(0) - 1, indices])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel_ref(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        # Mask this line because it gets the wrong result when IS_VARLEN is True and HEAD_FIRST is True
        # T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))

class TestTorchMethod:

    def chunk_local_cumsum_scalar(
        self,
        g: torch.Tensor,
        chunk_size: int,
        reverse: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        head_first: bool = False,
        output_dtype: torch.dtype | None = torch.float,
        ref: bool = False,
    ) -> torch.Tensor:
        if head_first:
            B, H, T = g.shape
        else:
            B, T, H = g.shape
        assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
            "chunk_size must be a power of 2"
        )
        BT = chunk_size
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
        )
        NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
        g_org, g = g, torch.zeros_like(g, dtype=output_dtype or g.dtype)
        if ref:
            grid = (NT, B * H)
            chunk_local_cumsum_scalar_kernel_ref[grid](
                g_org,
                g,
                cu_seqlens,
                chunk_indices,
                T=T,
                B=B,
                H=H,
                BT=BT,
                HEAD_FIRST=head_first,
                REVERSE=reverse,
                IS_VARLEN=cu_seqlens is not None,
            )
        else:
            chunk_local_cumsum_scalar_kernel(
                g_org,
                g,
                cu_seqlens,
                chunk_indices,
                T=T,
                B=B,
                H=H,
                BT=BT,
                head_first=head_first,
                reverse=reverse,
                NT=NT,
            )
        return g

    @pytest.mark.parametrize("seed", [123, 356, 478])
    @pytest.mark.parametrize("dtype",
                             [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("batch", [1, 2, 6])
    @pytest.mark.parametrize("token", [4, 8])
    @pytest.mark.parametrize("head", [32, 64])
    @pytest.mark.parametrize("chunk_size", [2, 4, 64])
    @pytest.mark.parametrize("is_varlen", [True, False])
    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("head_first", [True, False])
    @pytest.mark.parametrize("output_dtype", [torch.float32])
    def test_deepseek_scaling_rope(
        self,
        seed,
        dtype,
        batch,
        token,
        head,
        chunk_size,
        is_varlen,
        reverse,
        head_first,
        output_dtype,
    ):
        torch.manual_seed(seed)
        cu_seqlens = None
        if is_varlen:
            cu_seqlens = torch.Tensor([0, token]).to(device=DEVICE, dtype=torch.long)
        g = torch.randn(batch,
                                token,
                                head,
                                device=DEVICE).to(dtype)
        ref_out = self.chunk_local_cumsum_scalar(
            g, chunk_size, reverse, cu_seqlens, head_first, output_dtype, ref = True)
        out = self.chunk_local_cumsum_scalar(
            g, chunk_size, reverse, cu_seqlens, head_first, output_dtype, ref = False)
        torch.testing.assert_close(ref_out, out, atol=5e-3, rtol=1e-3)
