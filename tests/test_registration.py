# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

import unitree_g1_tracking_isaaclab.tasks  # noqa: F401


def test_task_registration() -> None:
    spec = gym.spec("Isaac-Tracking-Flat-G1-No-State-Estimation")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["default_agent"] == "rsl_rl"

