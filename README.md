# Unitree G1 Tracking — Isaac Lab

This is a small external Isaac Lab task for training or continuing the 29-action
`Unitree-G1-Tracking-No-State-Estimation` policy with Isaac Sim PhysX.

The policy contract is intentionally unchanged:

- actor observations: 154
- critic observations: 286
- actions: 29
- actor/critic MLP: 512, 256, 128 with ELU and observation normalization
- motion: `assets/motions/g1/halay_loop.npz`, starting at 4.18 s

The jump phase at source frames 59–101 keeps the original motion timing but uses
half-strength imitation rewards. Balance and fall terms remain fully active.

## Install

The project uses the local Isaac Lab checkout at `/home/forkon/IsaacLab`:

```bash
cd /home/forkon/unitree_g1_tracking_isaaclab
uv sync
```

## Train from scratch (recommended)

```bash
uv run isaaclab train \
  --rl_library rsl_rl \
  --task Isaac-Tracking-Flat-G1-No-State-Estimation \
  --num_envs 512 \
  --max_iterations 30001 \
  --run_name g1_tracking_physx_fresh \
  --visualizer none
```

## Continue from the copied MJLab checkpoint

`--max_iterations` is the number of additional iterations. The copied checkpoint
is `model_8000.pt`, so 22,001 additional iterations end at iteration 30,000.

```bash
cd /home/forkon/unitree_g1_tracking_isaaclab

uv run isaaclab train \
  --rl_library rsl_rl \
  --task Isaac-Tracking-Flat-G1-No-State-Estimation \
  --num_envs 512 \
  --checkpoint checkpoints/model_8000.pt \
  --max_iterations 22001 \
  --run_name g1_tracking_physx_finetune \
  --visualizer none
```

## Watch many robots in Isaac Sim

This opens the Omniverse Kit viewport and displays 256 parallel environments:

The environment nearest the center also shows the NPZ reference pose as a
semi-transparent cyan ghost skeleton. It does not affect policy observations.

```bash
uv run isaaclab play \
  --rl_library rsl_rl \
  --task Isaac-Tracking-Flat-G1-No-State-Estimation \
  --checkpoint checkpoints/model_8000.pt \
  --num_envs 256 \
  --visualizer kit
```

To train while watching all 256 robots in the Kit viewport:

```bash
uv run isaaclab train \
  --rl_library rsl_rl \
  --task Isaac-Tracking-Flat-G1-No-State-Estimation \
  --num_envs 256 \
  --max_iterations 30001 \
  --run_name g1_tracking_physx_visible \
  --visualizer kit
```

The network and optimizer state can be loaded, but changing simulators is still a
physics-domain transfer. Isaac Lab uses a USD-derived G1 model, so the resumed
policy may temporarily regress while adapting even though tensor dimensions and
joint ordering are preserved.

## Physical slope test

The task uses one collision-enabled PhysX platform per environment. Launch with
a fixed platform pitch (degrees) to test whether the policy reacts to a real
incline:

```bash
G1_GROUND_SLOPE_DEG=10 uv run isaaclab play \
  --rl_library rsl_rl \
  --task Isaac-Tracking-Flat-G1-No-State-Estimation \
  --checkpoint checkpoints/model_8500.pt \
  --num_envs 16 \
  --visualizer kit
```

The accepted range is `-30` to `30` degrees. Omit the environment variable for
the normal flat platform.
