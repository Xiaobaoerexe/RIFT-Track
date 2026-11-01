# Compact surface depth observer

The rejected global-scene direct decoder is replaced. The pinned runtime source
keeps its terrain encoder, full camera transform, 32 samples per full-resolution
ray, local surface feature queries and noise model. Only its image correction
U-Net is replaced by a distilled 16/32/48-channel network (source:32/64/96).
Output remains48×64 optical depth in metres,0.1–6m, invalid0. Production uses
FP32 geometry/refinement/noise with immutable map caches and recompute backward.
The policy, dynamics and value networks separately retain BF16.

This is a **new learned approximation**, not a numerically equivalent execution
of the original network. Its old-model VJP equivalence result remains reported
and is not relabelled as passing. The original fixed5% old-VJP gate was suitable
for unchanged-function precision/execution optimization; it is insufficient as
a physical accuracy test for a new observation model.

Validation covers jointly valid depth at0.1–2m,2–4m,4–6m (mean-error limits5,10,16cm),
with validity F1 checked independently. Label-valid errors andp90/p95 tails are
also preserved. RTX supervision covers original terrains; the new task boxes
add exact ray/AABB collision-geometry references, not fictitious RTX labels.
Gradient checks cover the new function's converged central differences,
analytical planar depth/VJP, local pose response, frozen parameters and exact
recompute/autograd agreement. Nonsmooth visibility/hole boundaries are reported
separately; the unchanged hard-forward/surrogate-backward convention is retained.

Offline fitting:

```
python -m go2_physx_cts.fast_depth.train --data RTX_BANKS --data NEW_TERRAIN_BANKS \
  --depth-package ORIGINAL_DEPTH_PACKAGE --output CANDIDATE \
  --width 16 --steps 30000 --batch 96 --views-per-scene 64 --lr 0.0003
```

`generate_bank` uses the actual production map bundle and explicit train/val/test
layout splits. Generated source-model labels are marked`frozen_teacher`,neverRTX.
Training keeps source geometry/noise frozen and fits depth, validity, image
edges, CNN input sensitivities, available RTX labels and exact production-box
ray intersections. The released lineage used30k initial updates followed by
approximately10k updates with the added physical box labels. Cached training feature
tensors are temporary offline data; active locomotion always recomputes depth at
predicted poses and retains their derivatives.

`evaluate` records the strict old-source comparison, all terrain/layout/pose
groups, true RTX metrics and recompute invariance. `qualify` combines that report
with independent numerical/analytic and new-box audits plus the complete
4096-environment smoke. Every report is tied to the exact candidate SHA256.
Only this final qualification may produce`accepted.pt`for production.
These finite checks support an engineering launch; they do not establish learned
locomotion capability or guarantee that model errors cannot be exploited.

```
python -m go2_physx_cts.fast_depth.qualify --weights CANDIDATE/best.pt \
  --images SOURCE_COMPARISON.json --gradient POSE_AUDIT.json \
  --boxes COLLISION_GEOMETRY_AUDIT.json --benchmark FULL_ROUND.json \
  --output FINAL_OBSERVER_DIRECTORY
```

The API rejects old direct-depth weights, unaccepted checkpoints, source package
changes and unaccepted precision profiles. It delegates scene caching and
render/recompute to the pinned original runtime implementation. The original
runtime training project, weights and historical experiment records are retained.
