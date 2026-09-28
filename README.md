# Unitree G1 Tracking — Isaac Lab

This is a small external Isaac Lab task for continuing the 29-action
`Unitree-G1-Tracking-No-State-Estimation` policy with the Newton/MJWarp backend.

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

## Continue from the copied MJLab checkpoint

`--max_iterations` is the number of additional iterations. The copied checkpoint
is `model_8000.pt`, so 22,001 additional iterations end at iteration 30,000.

```bash
cd /home/forkon/unitree_g1_tracking_isaaclab

uv run isaaclab train \
  --rl_library rsl_rl \
  --task Isaac-Tracking-Flat-G1-No-State-Estimation \
  --num_envs 1024 \
  --checkpoint checkpoints/model_8000.pt \
  --max_iterations 22001 \
  --run_name g1_tracking_mjlab_continuation
```

## Play

The Newton visualizer does not require Isaac Sim or Omniverse Kit:

```bash
uv run isaaclab play \
  --rl_library rsl_rl \
  --task Isaac-Tracking-Flat-G1-No-State-Estimation \
  --checkpoint checkpoints/model_8000.pt \
  --num_envs 1 \
  --visualizer newton
```

The network and optimizer state can be loaded, but changing simulators is still a
physics-domain transfer. Isaac Lab uses a USD-derived G1 model, so the resumed
policy may temporarily regress while adapting even though tensor dimensions and
joint ordering are preserved.

