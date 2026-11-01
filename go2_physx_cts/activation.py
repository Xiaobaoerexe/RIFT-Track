"""Selective activation storage for the existing imagined physics/policy step.

This changes the save/recompute boundary, not the step or its gradients. Place
``matrix_checkpoint(core)`` INSIDE ``torch.compile`` so AOTAutograd can see the
policy on the core's ATen operations. Wrapping an already compiled opaque call
does not provide that guarantee. Image/depth rendering remains outside the core.
"""
import functools
import torch
from torch.utils.checkpoint import (
    CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts,
)


def _matrix_policy(context,op,*args,**kwargs):
    # Preserve costly hidden-layer products, while cheap pose/FK3x3 products,
    # small output heads, convolutions and dense terrain-query intermediates are
    # recomputed. Dimensions are architecture constants, independent of inputs.
    if op in (torch.ops.aten.mm.default,torch.ops.aten.bmm.default):
        left,right=args[:2]
        if left.shape[-1]>=64 and right.shape[-1]>=64:
            return CheckpointPolicy.MUST_SAVE
    elif op==torch.ops.aten.addmm.default:
        left,right=args[1:3]
        # GRUCell256 produces768-wide projections (three256-wide gates). Its eager native
        # implementation uses unsafe_split views and in-place gate activations;
        # those aliases can mutate an SAC-cached addmm result without the usual
        # view-version check detecting it. Recompute these small projections.
        # AOT functionalization may avoid the alias, but both execution modes
        # must retain the same recurrence gradients.
        if right.shape[-1]==3*256:
            return CheckpointPolicy.PREFER_RECOMPUTE
        if left.shape[-1]>=64 and right.shape[-1]>=64:
            return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def matrix_contexts():
    # Fresh per-call storage, with mutation checking left enabled. Nothing is
    # cached across trajectory steps, minibatches or optimizer updates.
    return create_selective_checkpoint_contexts(_matrix_policy)


def matrix_checkpoint(core):
    """Return a callable to compile, retaining the core's normal no-grad path.

    The surrounding command/noise/history update should run directly rather
    than inside a second, whole-step checkpoint. It is small, and nesting the
    old outer checkpoint would defeat the selective storage benefit.
    """
    @functools.wraps(core)
    def run(*args,**kwargs):
        if not torch.is_grad_enabled():
            return core(*args,**kwargs)
        return checkpoint(core,*args,use_reentrant=False,preserve_rng_state=True,
                          context_fn=matrix_contexts,**kwargs)
    return run
