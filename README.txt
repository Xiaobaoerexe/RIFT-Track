AS2 Depth CTS / SuperMimic-style physical and visual-latent imagination

2026-11-15 revision. Four configurations share one implementation:
175: variant=moe, three dense action experts, four independent task critics.
125: variant=events (historical config name only), plain Actor, four task critics.
156: variant=simple, three dense experts, one reward and one critic.
137: code-only copy of175, latent_actor_stop_gradient=true; do not launch or overwrite the existing137 experiment.
No variant contains a separate event-value channel or root event-Q objective.

Primary baseline: AS2 blind2061be6b0d592afe9e81223206cf5f864e45b9f4. UPSTREAM.json fixes source versions and distinguishes adaptation from source facts.

Shared training
Isaac Sim/PhysX, AS2 USD, IdealPD, raw12-joint action, no tanh. Target height0.36m; default thigh0.708360615872145, calf-1.41672123174429, mirrored hips+/-0.1. Initial release height0.4m. Load0..2kg per episode.
4096 environments,1real step/iteration,1000Replay steps/environment.8192x32 imagined transitions:6144teacher+2048student; objective=teacher mean+student mean. Execution microbatches preserve these denominators and one Actor optimizer step.
WM1024x3,2048x8 autoregressive windows,2updates before2500 and4after. Critic5passes x4minibatches=20optimizer steps total, not20per task. CTS2x160x64 supervised times with16stepburn-in;20480valid supervised frames/iteration.
120000iterations. AS2 bounded measured-window learning-rate schedule retained; no PPO KL or one-step candidate selection.

Scenes and rewards
After5000:2458ordinary tracking,328fast running,327abrupt turning,983crossing (328gap/328up/327down). Before2500flat only;2500..5000original terrain recipe. Recovery and avoidance are absent.
For156 these are sampling categories, not hidden policy tasks: one motion reward and one critic; no success reward or velocity grid. All three share20sordinary/8scrossing episode durations;156reads the calibration file only for this shared timing and geometry, not event rewards.125/175 use four task rewards and per-task/per-terrain grids.
Ordinary tracking forced-pure-forward probability0.10; retain source standing/turning sampling. Abrupt commands change every1s; tracking width relaxes for0.3s proportional to command change, then returns to the original kernel. Crossing height coefficient-1.
Crossing success: hard stable16ticks for up/down and32ticks for gap; once-only bonus, then continue3D motion. Pre-success pure-forward feasible approach commands; target corridor has no upper-forward boundary. Soft imagined completion is a run-length mass recurrence, bounded at1. It is a project surrogate, not a physical probability calibration.
Actor uses32steps differentiable total reward plus terminal total future value. Four critics have independent MLPs/normalizers and no policy latent input. Termination removes continuation; ordinary time limits bootstrap; crossing finite deadlines do not.
2026-11-16: removed value-range-triggered stopping, its bounds, counters and special stop-checkpoint path after it interrupted Critic prewarm. Critic values and targets are not clipped to reward-derived bounds. Existing nonfinite-update checks remain.

Perception and gradient ownership
Camera50Hz,48x64 metric optical depth,.1..6m. Mount relative to base_link=(.101958,0,.327)m, horizontal forward, nominal87x58degFOV; user confirmed2026-11-15. These are model coordinates, not measured AS2 hardware calibration.
Real depth: GPU first-hit rays against the exact static terrain collision mesh, followed by the frozen supplied keyed noise model. No learned geometry generator or imagined future image renderer remains in the active training path. Robot self-occlusion is excluded; the fixed AS2 head mesh lies below this camera field of view.
Every control step WM predicts next128Dvisual latent, with8step supervision. During imagination only, the student reads z0,z5,z10,... and holds that feature for five control steps (10Hz); WM always consumes its current50Hzlatent, and proprioception/GRU update every step. Holding does not detach gradients. Real collection,CTS burn-in/distillation and native evaluation remain50Hz. Real immutable final noisyFP16images may be CPU-cached; latent targets are re-encoded by the current CNN. Old cached images never substitute for future imagined observations.
Teacher takes privileged state and forward height map, no taskID. Student retains8expert proprioMoE450->32, depthCNN128,GRU256,fusionLinear+LayerNorm32.
Teacher+Actor share Adam. WM+CNN+reconstruction decoder share Adam. Student proprioMoE/GRU/fusion have the CTS optimizer. Critic has its own optimizer. Actor optimization freezes WM/Critic/student parameters while preserving their input derivatives.
137only: predicted latent is detached at student policy entry. WM latent recurrence and action->physical-reward derivatives remain connected. This is the sole intended algorithm difference from175.

Run
python -m go2_physx_cts.train --config <host-json> --output <run-directory> --usd <assets/as2/as2.usd>
Each run records actual loaded physics, resolved config, code commit, depth weights and runtime dependencies. See RELEASE_ACCEPTANCE.json for measured release evidence when populated.
No new locomotion capability or official MuJoCo visual-chain success is claimed by a smoke test. Fixed teacher/student native tests remain distinct from curriculum statistics. TRANSFER_ADAPTATION.txt defines the0.36m transfer/scoring contract and remaining integration boundary.

Latest visual timing contract: VISUAL_CLOCK_ACCEPTANCE.json supersedes the initial50Hzpolicy throughput. 156 uses a2GiB re-renderable CPU depth cache after a4GiB-cache run exhausted host RAM/swap; Replay samples and update counts are unchanged.

2026-11-16 speed release (175/125/156 only; existing137 untouched)
WM still recurs/supervises physics on2048x8 windows. Each optimizer update uniformly selects512whole windows for all8valid50Hz visual labels; initial latents use all2048windows, clean reconstruction keeps512images. Full-label validation is not subsampled. This is a measured project adaptation, not an Instinct default.
Cache stores final keyed noisyFP16observations and offsets slots by episode; hits avoid repeated noise inference. Frozen noise CNN usesBF16 withFP32ray geometry, random fields and corruption arithmetic. Trainable features are always recomputed.
CTS batches shared-Actor evaluations across the sequence, preservingGRU resets,masks,loss denominators and2optimizer updates. Critic groups unchanged minibatches by task, usesBF16hidden layers withFP32scalar heads/targets/Adam state, and still performs20steps. CUDA optimizers use fusedAdam. Unused Critic-fit feature graphs are omitted; all actual bootstrap-boundary input derivatives remain.
Image encoding chunks2048;Critic execution chunks65536. Bounded2compiler workers protect hostRAM. Per-host execution choices and measured tradeoffs are recorded inSPEED_ACCEPTANCE.json andSPEED_OPTIMIZATION.zh.txt. The prior visual-clock documents retain timing-contract history; this speed record supersedes their throughput numbers.
Every5000iterations, existing held-out model validation also records8/16/32step full-label errors inmodel_errors_*.json, preserving RNG and optimizer state. These reports neither stop training nor change learning rates. Value-range stopping remains deleted.
