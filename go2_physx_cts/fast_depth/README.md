# Direct-depth candidate

This is an offline-trained observation-model replacement candidate. It does not
change the physical world model, policy, rewards, rollout length, image size, or
frozen sensor-noise network. No accepted weights are included.

The frozen source terrain CNN produces map/local/context features. A trainable
CNN compresses those into a 1024-dimensional scene embedding; after fitting and
freezing, the embedding is cached per scene. Full camera position and rotation
condition a small MLP and three image upsampling blocks. They predict 48×64
optical-axis depth (0.1–6 m) and validity without per-pixel ray marching. The
unchanged source noise network uses the original explicit noise keys, hole odds
multiplier 2, and hard-forward/surrogate-backward convention. Model parameters:
3,518,466, including the scene compressor.

This compression can lose narrow gaps and obstacle edges. It must pass the
independent holdout; a speed improvement alone does not justify replacement.

## Commands

Run from a directory where `go2_physx_cts` is importable. `DEPTH_PACKAGE` is the
directory containing `go2_depth_input`, and `RTX_DATA` contains completed
`train_*`, `val_*`, `test_*` banks.

```bash
python -m go2_physx_cts.fast_depth.generate_bank \
  --maps MAP_BUNDLE.pt --depth-package DEPTH_PACKAGE --output TEACHER_BANKS \
  --batch 16 --train-views 512 --holdout-views 128

python -m go2_physx_cts.fast_depth.train \
  --data RTX_DATA --data TEACHER_BANKS --depth-package DEPTH_PACKAGE \
  --output CANDIDATE --steps 100 --batch 192 --vjp-every 4 --vjp-batch 4

# Choose a longer update budget only after inspecting measured first-100 timing.
python -m go2_physx_cts.fast_depth.train \
  --data RTX_DATA --data TEACHER_BANKS --depth-package DEPTH_PACKAGE \
  --output CANDIDATE --resume CANDIDATE/latest.pt --steps TOTAL_UPDATES \
  --batch 192 --vjp-every 4 --vjp-batch 4

python -m go2_physx_cts.fast_depth.evaluate \
  --weights CANDIDATE/best.pt --data RTX_DATA --data TEACHER_BANKS \
  --depth-package DEPTH_PACKAGE --output ACCEPTANCE \
  --batch 8 --views-per-scene 16 --gradient-views-per-scene 4 --probes 2
```

`generate_bank` accepts `maps: FloatTensor[S,512,512]` and a matching `metadata`
list with `name`, `split`, `layout_id`, and `difficulty`. Optional
`body_spawn_local` biases some viewpoints around the production start. Data
remain labelled `frozen_teacher`; this is compatibility supervision, **not RTX
ground truth** for the new terrains. Layout splits must be prepared before
calling this command. The generator refuses to overwrite an existing bank.

Training uses metric/edge losses, balanced validity supervision, finite pose
response supervision, and direct Sobolev VJP distillation. Frozen source CNN
features may be cached; trainable scene embeddings are recomputed every update.
Training and validation layout identities are checked for overlap. The
checkpoint records training identities for the independent test-split audit.

## Acceptance and integration

Acceptance reports image metrics and pose derivatives by terrain, layout, and
orientation family. Compatibility with the original observer must satisfy:

- Common-valid depth MAE ≤ 0.02 m, p95 ≤ 0.10 m; validity F1 ≥ 0.98.
- Edge MAE ≤ 0.05 m.
- Translation and SO(3) tangent VJP relative L2 ≤ 5%, for both geometry and
  geometry followed by frozen noise, with identical per-observation noise keys.
- RTX label-valid depth error no more than 10% above the old model on the same
  views; mask F1 may fall by at most 0.01, and edge error must not exceed the
  larger of 0.05m and 1.1 times the source error. These relative RTX checks avoid
  rejecting an exact source copy merely because the source itself is imperfect.
- New indoor/up/down obstacle terrain and nominal/transition/fallen/full
  orientation coverage, frozen parameters, and recompute/autograd equivalence.

The gate reports missing absolute RTX coverage rather than inventing evidence.
The VJP gate certifies agreement with the pinned old model's **surrogate**
derivatives. It does not make binary visibility or holes physically smooth.
The source package must retain weight SHA256
`137f90824abc604c03e19a3f15edc232f168c64bb9d1c47930d48bfe6260d8ae` for this
candidate's intended lineage; each checkpoint records the actual source hash.

Only a passing evaluator exports `accepted.pt`. FP32 and BF16 must be accepted
separately; a passed FP32 profile cannot authorize BF16. In deployment:

```python
from go2_physx_cts.fast_depth.api import FastDepthGenerator

engine = FastDepthGenerator(
    'ACCEPTANCE/accepted.pt', 'DEPTH_PACKAGE',
    device='cuda:0', chunk_size=256, precision='fp32',
    compile_mode='default', require_accepted=True,
)
cache = engine.prepare_scenes(maps, origins)
depth = engine.render(cache, camera_position, world_from_optical, scene_ids,
                      noise_keys, mode='recompute')
```

The API has `render_from_body`, `warmup`, and `inference`/`autograd` modes, keeps
the full rotation contract, freezes all parameters, and retains first-order
pose derivatives. Recompute explicitly does not support second-order policy
gradients. Default/off compilation avoids CUDA-graph output-aliasing assumptions.

## Local interface validation

On 2026-11-02, Python 3.11 / PyTorch 2.5.1 CPU checks passed syntax, direct
forward, finite mixed parameter/pose derivatives, bank generation, one training
update including Sobolev supervision, checkpoint restore into the frozen API,
and independent evaluator execution. Recompute versus ordinary autograd had
zero output and VJP difference in the small CPU sample. The untrained candidate
correctly failed quality/coverage acceptance and produced no accepted weights.
These are interface checks, not GPU throughput or observation-quality evidence.
