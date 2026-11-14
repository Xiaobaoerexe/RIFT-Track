AS2 Depth CTS / SuperMimic-style learned physical and visual-latent imagination

Three controlled experiments share one implementation, selected by cfg.variant:
137, ~/Documents, 48GB: simple — plain Actor, one motion reward/value, no velocity grid or success reward.
125, ~/motion_control, 24GB: events — same Actor, five task rewards, velocity grids and hard-event value learning.
175, ~/Admin, 48GB: moe — events configuration plus three dense action experts.

The engineering baseline is the actual137 AS2 blind source2061be6b0d592afe9e81223206cf5f864e45b9f4.
UPSTREAM.json records pinned external references and what is adapted. Original blind/depth projects and historical evidence are outside this isolated tree and are not deleted.

Common training contract
Isaac Sim/PhysX; AS2 USD and IdealPD; raw12-joint actions with scale.3125 and no tanh.
4096 real environments, one real step per iteration, Replay1000 steps/environment.
8192 imagined trajectories=6144 teacher+2048 student, horizon32, exact teacher mean+student mean Actor objective.
WM1024x3, batch2048, autoregressive8 steps; two updates before2500, four afterwards.
AS2 Critic fitting: five passes/four minibatches; execution microbatches do not change sample membership or optimizer steps.
Visual CTS:2x160 sequences,64 loss steps+16 burn-in, equivalent20480 supervised frames/iteration; adaptation from blind20x1024 independent histories, not a claim of identical temporal learning.
120000 iterations, no6/8-hour cap. Actual GPU memory/throughput must be measured separately on each host before starting.

Perception and ownership
Teacher sees proprioceptive privileged input and a forward height map; no taskID input.
Student keeps8-expert proprioception MoE450->32, depth CNN128, GRU256, Linear+LayerNorm fusion32.
Depth CNN has one optimizer owner: world-model reconstruction. Student MoE/GRU/fusion receive latent+action distillation only.
Teacher+shared Actor share one Adam; world model+depth CNN share one Adam; student remainder and Critic each have their own Adam.
Frozen depth/noise models generate held5Hz captures. Imagination starts from an encoded actual capture, then WM predicts128-D latent changes at capture refresh steps. It does not render future depth or reuse future real images.
WM predicts physical state/complete pose, contacts and actual AS2 reward telemetry; it never consumes a hidden taskID.

Reward/value split
simple uses exactly the fifteen AS2 motion terms, including its high-speed tracking-width blend and3k coefficient ramp. No success bonus is inserted.
events/moe preserve continuous long-term value and add a task-conditioned hard-event channel in the same Critic trunk. Actor combines continuous32-step BPTT+tail value with one root afterstate event value; it does not sum repeated success values along all32 steps.
Special approach commands start at0.6–1.0m/s for the8s deadline; the lowerbound is derived from the longestgap approach plus stability/startup time. Grid downgrades cannot make successful completion kinematically impossible.
Hard success/failure labels come from pre-reset PhysX transitions. No sigmoid-product completion, soft stable timer or soft cumulative success remains.
New event bonuses require complete-episode mature-AS2 policy calibration. Existing Go2 calibration numbers are invalid.

Run interface
python -m go2_physx_cts.train --config <resolved-json> --output <run-directory> --usd <assets/as2/as2.usd>
Use the runtime/version validation and complete-update GPU test implemented by tools/validate_training.py before formal runs. Host launch paths, checkpoint/depth hashes and measured throughput belong in each run manifest.
No valid new-model MuJoCo/real-robot claim is made here. Obsolete Go2 RoboGauge adapters were removed; native fixed-condition PhysX diagnosis remains separate from curriculum statistics.

Verification
python -m unittest discover -s tests
CPU invariants establish execution/algebra only. Loaded USD/PhysX parameter readback, full Actor backward, complete sustained updates and GPU memory/time acceptance remain required before launch.
