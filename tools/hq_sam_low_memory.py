"""Chunk HQ-SAM global attention queries without changing weights or prompts."""

from __future__ import annotations

import torch


def chunked_attention(self, x: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    from segment_anything.modeling.image_encoder import get_rel_pos

    batch, height, width, _ = x.shape
    tokens = height * width
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, -1)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, batch * self.num_heads, tokens, -1).unbind(0)
    output = torch.empty_like(q)
    if self.use_rel_pos:
        relative_h = get_rel_pos(height, height, self.rel_pos_h)
        relative_w = get_rel_pos(width, width, self.rel_pos_w)
    for start in range(0, tokens, chunk_size):
        end = min(tokens, start + chunk_size)
        query = q[:, start:end]
        attention = (query * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            indices = torch.arange(start, end, device=x.device)
            rh = torch.einsum("bqc,qkc->bqk", query, relative_h[indices // width])
            rw = torch.einsum("bqc,qkc->bqk", query, relative_w[indices % width])
            attention = attention.view(
                batch * self.num_heads, end - start, height, width
            )
            attention.add_(rh[..., None]).add_(rw[:, :, None, :])
            attention = attention.flatten(2)
        output[:, start:end] = attention.softmax(dim=-1) @ v
    output = output.view(batch, self.num_heads, height, width, -1)
    output = output.permute(0, 2, 3, 1, 4).reshape(batch, height, width, -1)
    return self.proj(output)


def enable_low_memory_attention(model) -> None:
    from types import MethodType

    for block in model.image_encoder.blocks:
        if block.window_size == 0:
            block.attn.forward = MethodType(chunked_attention, block.attn)
