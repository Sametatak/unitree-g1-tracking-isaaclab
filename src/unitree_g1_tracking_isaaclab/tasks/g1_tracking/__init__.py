# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="Isaac-Tracking-Flat-G1-No-State-Estimation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:G1TrackingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

