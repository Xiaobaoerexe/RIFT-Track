"""Detached Critic inputs: bounded blocks and lossless one-bit validity.

This cache never participates in Actor state derivatives. CUDA gathering uses
block address tables, so arbitrary minibatches keep their exact row ordering
without a Python loop per row or a multi-GiB contiguous allocation.
"""
import torch


def _cuda_kernels():
    # Keep CPU-only policy export independent of a Triton installation.
    import triton
    import triton.language as tl

    @triton.jit
    def pack(Crop, Height, Valid, N: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
        x = tl.program_id(0) * B + tl.arange(0, B)
        row, pixel = x // P, x % P
        h = tl.load(Crop + row * (2 * P) + pixel, x < N * P, 0)
        v = tl.load(Crop + row * (2 * P) + P + pixel, x < N * P, 0) != 0
        tl.store(Height + x, h, x < N * P)
        bits = tl.reshape(v.to(tl.int32), (B // 8, 8))
        weights = 1 << tl.arange(0, 8)
        byte = tl.sum(bits * weights[None, :], 1).to(tl.uint8)
        y = tl.program_id(0) * (B // 8) + tl.arange(0, B // 8)
        tl.store(Valid + y, byte, y < N * P // 8)

    @triton.jit
    def gather(Heights, Valids, Indices, Out, P: tl.constexpr,
               ROWS: tl.constexpr, HALF: tl.constexpr, B: tl.constexpr):
        dst = tl.program_id(0)
        source = tl.load(Indices + dst)
        block, row = source // ROWS, source % ROWS
        dtype = tl.bfloat16 if HALF else tl.float32
        hp = tl.load(Heights + block).to(tl.pointer_type(dtype))
        vp = tl.load(Valids + block).to(tl.pointer_type(tl.uint8))
        pixel = tl.program_id(1) * B + tl.arange(0, B)
        height = tl.load(hp + row * P + pixel, pixel < P, 0)
        packed = tl.load(vp + row * (P // 8) + pixel // 8, pixel < P, 0)
        valid = (packed >> (pixel % 8)) & 1
        tl.store(Out + dst * (2 * P) + pixel, height, pixel < P)
        tl.store(Out + dst * (2 * P) + P + pixel, valid, pixel < P)
    return pack, gather


_kernels = None


class CriticGeometryCache:
    def __init__(self, rows, device, dtype, block_rows=2048):
        self.rows, self.device, self.dtype = int(rows), torch.device(device), dtype
        self.block_rows = int(block_rows)
        self.heights, self.validity = [], []
        self._height_ptrs = self._valid_ptrs = None
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('Critic cache supports FP32 or BF16 heights')

    def append(self, crop):
        global _kernels
        if crop.requires_grad:
            raise ValueError('Critic geometry cache accepts only detached inputs')
        n = len(crop)
        if tuple(crop.shape[1:]) != (2, 64, 128) or n > self.block_rows:
            raise ValueError('Unexpected Critic geometry block')
        height = torch.empty((n, 64, 128), device=self.device, dtype=self.dtype)
        valid = torch.empty((n, 1024), device=self.device, dtype=torch.uint8)
        if self.device.type == 'cuda':
            if _kernels is None:
                _kernels = _cuda_kernels()
            _kernels[0][((n * 8192 + 1023) // 1024,)](crop.contiguous(), height, valid, n, 8192, 1024)
        else:
            height.copy_(crop[:, 0])
            bits = crop[:, 1].bool().reshape(n, 1024, 8).to(torch.int16)
            weights = 1 << torch.arange(8, device=self.device, dtype=torch.int16)
            valid.copy_((bits * weights).sum(-1).to(torch.uint8))
        self.heights.append(height)
        self.validity.append(valid)

    def finish(self):
        if sum(len(x) for x in self.heights) != self.rows:
            raise ValueError('Incomplete Critic cache')
        if any(len(x) != self.block_rows for x in self.heights[:-1]):
            raise ValueError('Only the last Critic block may be partial')
        if self.device.type == 'cuda':
            self._height_ptrs = torch.tensor([x.data_ptr() for x in self.heights], device=self.device, dtype=torch.int64)
            self._valid_ptrs = torch.tensor([x.data_ptr() for x in self.validity], device=self.device, dtype=torch.int64)
        return self

    def gather(self, indices):
        indices = indices.reshape(-1).contiguous()
        out = torch.empty((len(indices), 2, 64, 128), device=self.device, dtype=self.dtype)
        if not len(indices):
            return out
        if self.device.type == 'cuda':
            _kernels[1][(len(indices), 8)](self._height_ptrs, self._valid_ptrs, indices, out,
                8192, self.block_rows, self.dtype == torch.bfloat16, 1024)
        else:
            # CPU export/validation path groups rows by storage block.
            for block, (heights, valid) in enumerate(zip(self.heights, self.validity)):
                dst = ((indices // self.block_rows) == block).nonzero().flatten()
                src = indices[dst] % self.block_rows
                out[dst, 0] = heights[src]
                bits = (valid[src, :, None] >> torch.arange(8, device=self.device)) & 1
                out[dst, 1] = bits.reshape(-1, 64, 128).to(self.dtype)
        return out

    @property
    def storage_bytes(self):
        return sum(x.numel() * x.element_size() for x in self.heights + self.validity)
