Go2 PhysX CTS / source-aligned SuperMimic port

Implementation reference: https://github.com/Zhengtianjin/go2-supermimic
Pinned commit: d9eee9b3f248e5300e636c433255db2c1c3e086a
UPSTREAM.json binds the downloaded source files to verified Git blob SHA1 and SHA256.

This is an independent implementation. It does not import the retired Newton
SMCTS package. Required physics is Isaac Lab3 with Isaac Sim6.2 / PhysX. The
reference repository uses mjlab/MuJoCo Warp; its physics results do not certify
this port.

Training: python -m go2_physx_cts.supervise --output RUN_DIRECTORY --usd GO2_USD
          --driver OFFICIAL_MOE_TORCHSCRIPT --xml MUJOCO_FLAT_XML
All paths must identify real assets. The supervisor uses an exclusive file lock.
Flat training starts cold. A terrain phase can start only after two consecutive
MuJoCo flat velocity checks pass; its second online expert is the resulting flat
checkpoint. No upstream chase policy is presented as a velocity skill expert.

Mechanism checks: python -m unittest discover -s tests -v
Physical/update smoke: python -m go2_physx_cts.train --config CONFIG_JSON
                      --output SCRATCH --usd GO2_USD --smoke
Official/standing positive control: same command plus --probe-only.

The production contract is4096 envs, collect1, actor1, replay1000/env,
8192 imagined starts x32steps, WM2048 windows x8steps,2 WM updates on flat and4
on terrain. Teacher/student imagination allocation is6144/2048 with one full
batch mean (0.75/0.25), no extra averaging by branch. Both sample all providers.
Only teacher encoder and shared actor are optimized by imagined return. The
student encoder receives latent distillation on real student-source histories.
The frozen critic has physical inputs, not actor-owned latent inputs. Freezing
parameters does not cut state/action input gradients.

Replay contains every provider's real executed actions and episode context.
There is no expert-action cloning loss and no separate WM-only reference pool.
Flat online provider: mixed-terrain MoE10%; source flat default has no driver.
Terrain recipe: mixed-terrain MoE10% plus this task's validated flat expert10%.
The latter provides velocity skills, not demonstrated jumping/ducking skills.

Task adaptations: body-frame(vx,vy,wz) commands, blind45x10 CTS history,
167-D privileged teacher,169-D physical/branch critic,70-D model state,
176-D WM input /73-D output. The fifth context entry is fixed0..1kg rigid payload
at the original base COM, sampled only at episode reset. Actions are linear,
unsquashed12-D joint targets with source PD/motor scales. The deployed export
requires the corresponding default pose, scales and PD in physics_contract.json.

The nominal contact coefficient is1 and COM randomization is zero in this first
port. Source sliding/spinning/rolling friction and COM randomization are not
claimed reproduced: PhysX contact parameters differ from MuJoCo. Encoder-bias
randomization is also not claimed. The terrain recipe's sizes and proportions
are ported to Isaac Lab generators; its triangles are not byte-identical mjlab
meshes. A differentiable height map is raycast from the actual PhysX mesh.

The source's WM, autoregressive loss, TD(lambda), soft survival, rare-window
selection/archive, running normalization, OU noise, actor LR ramp, optimizer
settings and clips are reproduced. CTS creates an additional20x1024 student
update schedule. Actual timing includes it.

Formal training has no wall-time cutoff.40k iterations is the source-default
review budget. Every5000 iterations MuJoCo evaluates both branches and0/1kg
loads. Improvement can extend the review by20k; measured degradation can roll
policy/critic back while retaining live WM/data. Repeated severe failure stops
for inspection. The terrain phase currently uses a flat MuJoCo transfer sentinel;
this does not certify stair/terrain success. Physical training fall metrics also
drive the source-style collapse guard.

Checkpoints include all four models/optimizers, normalization and RNG metadata.
Like the source, resume starts new simulator episodes and fresh replay; it is
not a bit-identical full-simulator/replay continuation. Old checkpoints are
intentionally incompatible. No training capability or B0 is claimed by smoke.

MuJoCo diagnostics bind each rigid body's mass, COM and full inertia tensor to
the actual training USD using assets/go2_usd_inertials.json. Fixed feet/heads
are aggregated with rigid transforms and the parallel-axis theorem. Nominal
mass is16.08700025kg; the unmodified legacy XML was15.206408kg. Zero-joint
body frames were checked against USD. Collision shapes and contact solvers
remain platform-specific, so inertial alignment does not imply identical physics.
