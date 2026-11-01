# Blind locomotion revision, 2026-11-02

Changes made on server137; transferred directly to server175.
Policy/CTS/Critic architectures,4096envs,8192x32 imagination,
CTS20x1024 and Critic5x4 updates retained. Equal branch means;
6144teacher/2048student samples unchanged.

Go2RLLab reference:6c95d9feef4814c25eb6536e61bb38e1d1d718d4.
Replace extra upright/crouch/hard-landing/termination rewards with official15
PhysX terms. Height/vz ramps finish3000(project timing adaptation).
Keep robot-specific assets/control; As2 height target remains0.30m.

WM trunk1024x1024x1024 and state70/input176 unchanged. Output73->113:
torque12, acceleration12, foot horizontal velocity8, log1p limb force8.
Labels retain terminal pre-reset transitions. Real contact>5N; sigmoid in
log-force space only in BPTT, not exact differentiation of rigid collision.

Warmup0-1000; Critic1000-1500; Actor ramp1500-2500.
Flat until2500; flat40%/three slopes20% each until5000; original ten
terrains after5000. Original terrain promotion/demotion retained.
Speed stages5000/10000/20000/30000/35000; original ranges/caps retained.
No new task/depth/action-MoE/grid module.

Initial WM/Actor/Critic LR2e-4; CTS3e-4. No cosine decay.
After5000, every1000 rounds compare bounded /1.3,1,x1.3 rates.
Supervised validation for WM/Critic/CTS; real PhysX probes also for Actor.
Trial parameters/optimizer/RNG restored; only selected LR persists.
CTS adaptation after three stable fixed-native teacher probes.
Native-only CTS gate explicitly differs from depth native+transfer gate.
Original collapse rollback retained.

Go2 mixed official provider10% plus retained project flat expert10%, revalidated before launch.
As2 has no verified expert; never substitute Go2 policy or claim
verified As2 RoboGauge transfer. Native diagnostics remain available.

Each robot120000 iterations, no hour cap. Start As2 only after Go2 exits
successfully at full budget and creates its student export.
Failure/signal stops sequence. Engineering checks are not gait capability.
