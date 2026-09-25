"""
One-shot all-reduce in Triton on top of PyTorch Symmetric Memory.

Reference: https://docs.pytorch.org/docs/2.14/symmetric_memory.html

Symmetric memory gives every rank a raw pointer to every peer's buffer, so a
Triton kernel can just `tl.load` straight out of another GPU's HBM over the
fabric (xGMI / NVLink).  A "one-shot" all-reduce is then the simplest possible
algorithm:

    barrier -> every rank reads all N peer buffers and sums them -> barrier

Every rank reads the whole input from every peer, so it moves (world_size - 1)
* numel bytes per GPU.  That is more traffic than ring/two-shot, but it is a
single kernel with no intermediate hops, which wins on latency for small
tensors -- the regime where NCCL/RCCL is dominated by launch and handshake
overhead.

The barrier is written in portable Triton atomics (not inline PTX like the
upstream `ptx_utils.symm_mem_sync`) so it runs on both CUDA and ROCm.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Device-side barrier over the symmetric-memory signal pads
# ---------------------------------------------------------------------------


@triton.jit
def _symm_mem_barrier(
    signal_pad_ptrs,  # int64[world_size]: pointer to each rank's signal pad
    rank: tl.constexpr,
    world_size: tl.constexpr,
    RELEASE: tl.constexpr,  # publish my writes before signalling
    ACQUIRE: tl.constexpr,  # make peer writes visible after waiting
):
    """Block-scoped barrier across all ranks.

    Signal pads are viewed as a uint32 matrix of shape [num_blocks, world_size].
    Block `pid` on rank `r` announces itself by flipping slot [pid, r] on every
    peer's pad, then waits until every slot in its own row [pid, :] has been
    flipped.  Because the waiter *consumes* the flag (CAS 1 -> 0) the pad is
    left zeroed, so the same pad can be reused by the next barrier.

    The send side spins on CAS 0 -> 1 rather than doing a plain store: a rank
    that races ahead into the next barrier must not clobber a flag the peer has
    not consumed yet, or that peer would wait forever.
    """
    pid = tl.program_id(0)

    # --- announce arrival to every peer -----------------------------------
    for peer in tl.static_range(world_size):
        remote_pad = tl.load(signal_pad_ptrs + peer).to(tl.pointer_type(tl.uint32))
        slot = remote_pad + pid * world_size + rank
        while tl.atomic_cas(slot, 0, 1, sem=RELEASE, scope="sys") != 0:
            pass

    # --- wait for every peer ----------------------------------------------
    local_pad = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))
    for peer in tl.static_range(world_size):
        slot = local_pad + pid * world_size + peer
        while tl.atomic_cas(slot, 1, 0, sem=ACQUIRE, scope="sys") != 1:
            pass


# ---------------------------------------------------------------------------
# One-shot all-reduce kernel
# ---------------------------------------------------------------------------


@triton.jit
def one_shot_all_reduce_kernel(
    buffer_ptrs,  # int64[world_size]: pointer to each rank's symmetric buffer
    signal_pad_ptrs,  # int64[world_size]
    output_ptr,
    numel,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Everyone's contribution must be in place before anyone starts reading.
    _symm_mem_barrier(
        signal_pad_ptrs, rank, world_size, RELEASE="release", ACQUIRE="acquire"
    )

    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # Grid-stride: the grid is capped by the signal-pad capacity (the barrier
    # needs one uint32 per (block, peer) pair), so a block may own several
    # chunks of a large tensor.
    for offset in tl.range(pid * BLOCK_SIZE, numel, num_pids * BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in tl.static_range(world_size):
            # Peer `i`'s buffer lives on another GPU; the load goes over the
            # interconnect and is indexed exactly like a local tensor.
            buffer_rank = tl.load(buffer_ptrs + i).to(
                tl.pointer_type(output_ptr.dtype.element_ty)
            )
            x = tl.load(buffer_rank + offsets, mask=mask, other=0.0)
            acc += x.to(tl.float32)

        tl.store(output_ptr + offsets, acc.to(output_ptr.dtype.element_ty), mask=mask)

    # Nobody may overwrite their input buffer until all peers finished reading it.
    _symm_mem_barrier(
        signal_pad_ptrs, rank, world_size, RELEASE="release", ACQUIRE="acquire"
    )


# ---------------------------------------------------------------------------
# Host-side wrapper
# ---------------------------------------------------------------------------

_PTR_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _peer_ptrs(hdl, device) -> tuple[torch.Tensor, torch.Tensor]:
    """int64 device tensors holding the peer buffer / signal-pad pointers.

    Keyed on the pointer values themselves, not on `id(hdl)`: handles are
    recreated per call and CPython recycles object ids, so an id-keyed cache
    happily hands back pointers into a freed symmetric allocation.  Keying on
    the addresses is always sound because the cached value is derived purely
    from them.
    """
    key = (tuple(hdl.buffer_ptrs), tuple(hdl.signal_pad_ptrs))
    if key not in _PTR_CACHE:
        _PTR_CACHE[key] = (
            torch.tensor(key[0], dtype=torch.int64, device=device),
            torch.tensor(key[1], dtype=torch.int64, device=device),
        )
    return _PTR_CACHE[key]


def one_shot_all_reduce(
    inp: torch.Tensor,
    group_name: str = "0",
    out: torch.Tensor | None = None,
    BLOCK_SIZE: int = 2048,
    num_warps: int = 8,
) -> torch.Tensor:
    """Sum `inp` across the group.  `inp` must be a symmetric-memory tensor.

    Args:
        inp: 1D tensor allocated with `symm_mem.empty(...)`, identical shape and
            dtype on every rank.
        group_name: process group the tensor was rendezvous'd on.
        out: optional destination (ordinary tensor is fine).  Defaults to a
            fresh tensor.  Passing `out is inp` gives in-place semantics via a
            staging buffer -- see below.
    """
    assert inp.is_contiguous(), "input must be contiguous"
    hdl = symm_mem.rendezvous(inp, group_name)
    if out is None:
        out = torch.empty_like(inp)

    # Writing straight back into the symmetric buffer would race: block j on
    # this rank can overwrite our slice while block j on a *peer* is still
    # reading it, and the closing barrier is far too late to stop that.  Only a
    # barrier between the read and the write phases would make it safe, which
    # the grid-stride loop cannot do without buffering the whole tensor.  So
    # reduce into scratch and copy back once every peer has finished reading.
    aliased = out.data_ptr() == inp.data_ptr()
    dst = torch.empty_like(inp) if aliased else out

    buffer_ptrs, signal_pad_ptrs = _peer_ptrs(hdl, inp.device)

    # The barrier consumes one uint32 per (block, peer), so the grid cannot be
    # larger than the signal pad allows.
    max_blocks = hdl.signal_pad_size // 4 // hdl.world_size
    num_blocks = min(triton.cdiv(inp.numel(), BLOCK_SIZE), max_blocks)

    one_shot_all_reduce_kernel[(num_blocks, 1, 1)](
        buffer_ptrs,
        signal_pad_ptrs,
        dst,
        inp.numel(),
        rank=hdl.rank,
        world_size=hdl.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    if aliased:
        out.copy_(dst)
    return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group_name = dist.group.WORLD.group_name
    device = torch.device("cuda", local_rank)

    _log(rank, f"world_size={world_size}  backend={symm_mem.get_backend(device)}")
    _log(rank, f"visible devices = {os.environ.get('HIP_VISIBLE_DEVICES', 'all')}")

    dtype = torch.float32
    failures = 0

    for numel in (8, 1024, 4096, 100_000, 1 << 20, (1 << 20) + 17):
        # Each rank owns one shard of a logically replicated 1D tensor.  Rank r
        # contributes (r + 1) * arange(numel), so the exact all-reduced value is
        # sum_r (r+1) * i = world_size*(world_size+1)/2 * i.
        inp = symm_mem.empty(numel, dtype=dtype, device=device)
        inp.copy_(
            (rank + 1) * torch.arange(numel, dtype=dtype, device=device)
        )

        out = one_shot_all_reduce(inp, group_name)

        scale = world_size * (world_size + 1) // 2
        expected = scale * torch.arange(numel, dtype=dtype, device=device)
        ok = torch.equal(out, expected)

        # Cross-check against RCCL/NCCL on random data too.
        ref = torch.randn(numel, dtype=dtype, device=device)
        inp.copy_(ref)
        out2 = one_shot_all_reduce(inp, group_name)
        dist.all_reduce(ref)
        close = torch.allclose(out2, ref, rtol=1e-5, atol=1e-5)

        # Every rank must agree before we call it a pass.
        verdict = torch.tensor([ok and close], device=device, dtype=torch.int32)
        dist.all_reduce(verdict)
        passed = int(verdict.item()) == world_size
        failures += not passed
        _log(
            rank,
            f"  numel={numel:<9} exact={'ok' if ok else 'FAIL'} "
            f"vs-nccl={'ok' if close else 'FAIL'} "
            f"-> {'PASS' if passed else 'FAIL'}",
        )

    # bfloat16: the kernel accumulates in fp32 regardless of storage dtype, so
    # the result should be bit-identical to reducing in fp32 and rounding once.
    # (Comparing against NCCL's bf16 all-reduce would be the *weaker* check --
    # it rounds at every step of the ring and lands ~2 bf16 ulps away.)
    numel = 65536
    inp = symm_mem.empty(numel, dtype=torch.bfloat16, device=device)
    ref = torch.randn(numel, dtype=torch.bfloat16, device=device)
    inp.copy_(ref)
    out = one_shot_all_reduce(inp, group_name)
    ref32 = ref.float()
    dist.all_reduce(ref32)
    ok = torch.equal(out, ref32.bfloat16())
    verdict = torch.tensor([ok], device=device, dtype=torch.int32)
    dist.all_reduce(verdict)
    passed = int(verdict.item()) == world_size
    failures += not passed
    _log(rank, f"  bfloat16  numel={numel:<9} -> {'PASS' if passed else 'FAIL'}")

    # In-place variant: output aliases the symmetric input buffer.
    numel = 4096
    inp = symm_mem.empty(numel, dtype=dtype, device=device)
    inp.fill_(float(rank + 1))
    one_shot_all_reduce(inp, group_name, out=inp)
    expected_val = float(world_size * (world_size + 1) // 2)
    ok = torch.equal(inp, torch.full_like(inp, expected_val))
    verdict = torch.tensor([ok], device=device, dtype=torch.int32)
    dist.all_reduce(verdict)
    passed = int(verdict.item()) == world_size
    failures += not passed
    _log(rank, f"  in-place  numel={numel:<9} -> {'PASS' if passed else 'FAIL'}")

    dist.barrier()
    _log(rank, "ALL TESTS PASSED" if failures == 0 else f"{failures} FAILURE(S)")
    dist.destroy_process_group()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
