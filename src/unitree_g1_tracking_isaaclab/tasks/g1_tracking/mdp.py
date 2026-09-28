# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, cast

import isaaclab.envs.mdp as base_mdp
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Body order used when halay_loop.npz was produced in unitree_rl_mjlab.
SOURCE_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


class MotionLoader:
    """Load the existing MJLab NPZ and select bodies by its source-name order."""

    def __init__(self, motion_file: str, body_names: tuple[str, ...], device: str, start_time_s: float):
        data = np.load(motion_file)
        self.joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=device)
        source_body_ids = [SOURCE_BODY_NAMES.index(name) for name in body_names]
        self.body_pos_w = torch.as_tensor(data["body_pos_w"][:, source_body_ids], dtype=torch.float32, device=device)
        self.body_quat_w = torch.as_tensor(data["body_quat_w"][:, source_body_ids], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.as_tensor(
            data["body_lin_vel_w"][:, source_body_ids], dtype=torch.float32, device=device
        )
        self.body_ang_vel_w = torch.as_tensor(
            data["body_ang_vel_w"][:, source_body_ids], dtype=torch.float32, device=device
        )
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        self.start_frame = round(start_time_s * self.fps) % self.joint_pos.shape[0]
        if start_time_s < 0.0:
            raise ValueError("start_time_s must be non-negative.")
        if self.start_frame:
            for name in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                setattr(self, name, torch.roll(getattr(self, name), shifts=-self.start_frame, dims=0))
        self.time_step_total = self.joint_pos.shape[0]


class MotionCommand(CommandTerm):
    """Advance the cyclic reference and reset G1 close to its current frame."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        joint_ids, _ = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)
        body_ids, _ = self.robot.find_bodies(list(cfg.body_names), preserve_order=True)
        self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self.body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        self.robot_anchor_body_id = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_id = cfg.body_names.index(cfg.anchor_body_name)
        self.motion = MotionLoader(cfg.motion_file, cfg.body_names, self.device, cfg.start_time_s)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = self.body_pos_w.clone()
        self.body_quat_relative_w = self.body_quat_w.clone()
        self.metrics["anchor_pos_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.joint_pos, self.joint_vel), dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_id]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_id]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch[:, self.joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.torch[:, self.joint_ids]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.body_ids]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.body_ids]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.robot_anchor_body_id]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.robot_anchor_body_id]

    def _update_metrics(self) -> None:
        self.metrics["anchor_pos_error"][:] = torch.linalg.vector_norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.time_steps[env_ids] = 0
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_quat = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        pose_ranges = torch.tensor(
            [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=self.device,
        )
        pose_noise = sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), self.device)
        root_pos += pose_noise[:, :3]
        root_quat = quat_mul(
            quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]), root_quat
        )

        velocity_ranges = torch.tensor(
            [
                self.cfg.velocity_range.get(key, (0.0, 0.0))
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            device=self.device,
        )
        velocity_noise = sample_uniform(
            velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), self.device
        )
        root_lin_vel += velocity_noise[:, :3]
        root_ang_vel += velocity_noise[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, self.device)
        joint_limits = self.robot.data.soft_joint_pos_limits.torch[env_ids[:, None], self.joint_ids]
        joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])
        joint_vel = self.joint_vel[env_ids]

        self.robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat((root_pos, root_quat), dim=-1), env_ids=env_ids
        )
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=torch.cat((root_lin_vel, root_ang_vel), dim=-1), env_ids=env_ids
        )
        self.body_pos_relative_w[env_ids] = self.body_pos_w[env_ids]
        self.body_quat_relative_w[env_ids] = self.body_quat_w[env_ids]

    def _update_command(self) -> None:
        self.time_steps += 1
        wrapped = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if wrapped.numel() > 0:
            self._resample_command(wrapped)

        ref_anchor_pos = self.anchor_pos_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        ref_anchor_quat = self.anchor_quat_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].expand(-1, len(self.cfg.body_names), -1).clone()
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        robot_anchor_pos[..., 2] = ref_anchor_pos[..., 2]
        delta_quat = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_quat, self.body_quat_w)
        self.body_pos_relative_w = robot_anchor_pos + quat_apply(
            delta_quat, self.body_pos_w - ref_anchor_pos
        )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the cyclic motion command."""

    class_type: type[CommandTerm] = MotionCommand
    asset_name: str = MISSING
    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: tuple[str, ...] = MISSING
    joint_names: tuple[str, ...] = MISSING
    start_time_s: float = 0.0
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.03, 0.03)


def motion_anchor_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return pos


def motion_anchor_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    _, quat = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    matrix = matrix_from_quat(quat)
    return matrix[..., :2].reshape(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    count = len(command.cfg.body_names)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None].expand(-1, count, -1),
        command.robot_anchor_quat_w[:, None].expand(-1, count, -1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return pos.reshape(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    count = len(command.cfg.body_names)
    _, quat = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None].expand(-1, count, -1),
        command.robot_anchor_quat_w[:, None].expand(-1, count, -1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    matrix = matrix_from_quat(quat)
    return matrix[..., :2].reshape(env.num_envs, -1)


def _body_ids(command: MotionCommand, body_names: tuple[str, ...] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if body_names is None or name in body_names]


def _in_reference_phase(command: MotionCommand, start_s: float | None, end_s: float | None) -> torch.Tensor:
    if start_s is None or end_s is None:
        return torch.zeros_like(command.time_steps, dtype=torch.bool)
    source_frame = (command.time_steps + command.motion.start_frame) % command.motion.time_step_total
    phase_s = source_frame / command.motion.fps
    if start_s <= end_s:
        return (phase_s >= start_s) & (phase_s <= end_s)
    return (phase_s >= start_s) | (phase_s <= end_s)


def _phase_scale(
    reward: torch.Tensor,
    command: MotionCommand,
    phase_start_s: float | None,
    phase_end_s: float | None,
    phase_scale: float,
) -> torch.Tensor:
    return reward * torch.where(_in_reference_phase(command, phase_start_s, phase_end_s), phase_scale, 1.0)


def motion_anchor_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start_s: float | None = None,
    phase_end_s: float | None = None,
    phase_scale: float = 1.0,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return _phase_scale(torch.exp(-error / std**2), command, phase_start_s, phase_end_s, phase_scale)


def motion_anchor_orientation_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start_s: float | None = None,
    phase_end_s: float | None = None,
    phase_scale: float = 1.0,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return _phase_scale(torch.exp(-error / std**2), command, phase_start_s, phase_end_s, phase_scale)


def motion_body_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start_s: float | None = None,
    phase_end_s: float | None = None,
    phase_scale: float = 1.0,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(torch.square(command.body_pos_relative_w - command.robot_body_pos_w), dim=-1).mean(-1)
    return _phase_scale(torch.exp(-error / std**2), command, phase_start_s, phase_end_s, phase_scale)


def motion_body_orientation_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start_s: float | None = None,
    phase_end_s: float | None = None,
    phase_scale: float = 1.0,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w).square().mean(-1)
    return _phase_scale(torch.exp(-error / std**2), command, phase_start_s, phase_end_s, phase_scale)


def motion_body_linear_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start_s: float | None = None,
    phase_end_s: float | None = None,
    phase_scale: float = 1.0,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(torch.square(command.body_lin_vel_w - command.robot_body_lin_vel_w), dim=-1).mean(-1)
    return _phase_scale(torch.exp(-error / std**2), command, phase_start_s, phase_end_s, phase_scale)


def motion_body_angular_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start_s: float | None = None,
    phase_end_s: float | None = None,
    phase_scale: float = 1.0,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(torch.square(command.body_ang_vel_w - command.robot_body_ang_vel_w), dim=-1).mean(-1)
    return _phase_scale(torch.exp(-error / std**2), command, phase_start_s, phase_end_s, phase_scale)


def torso_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids].squeeze(1)
    gravity = torch.tensor((0.0, 0.0, -1.0), device=env.device).expand(env.num_envs, -1)
    projected = quat_apply_inverse(quat, gravity)
    return torch.sum(torch.square(projected[:, :2]), dim=-1)


def torso_angular_velocity_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angular_velocity = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids].squeeze(1)
    return torch.sum(torch.square(angular_velocity[:, :2]), dim=-1)


def bad_anchor_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    phase_threshold: float | None = None,
    phase_start_s: float | None = None,
    phase_end_s: float | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    if phase_threshold is None:
        return error > threshold
    in_phase = _in_reference_phase(command, phase_start_s, phase_end_s)
    active_threshold = torch.where(in_phase, phase_threshold, threshold)
    return error > active_threshold


def bad_anchor_orientation(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    gravity = torch.tensor((0.0, 0.0, -1.0), device=env.device).expand(env.num_envs, -1)
    reference = quat_apply_inverse(command.anchor_quat_w, gravity)
    actual = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
    return torch.abs(reference[:, 2] - actual[:, 2]) > threshold


def bad_end_effector_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...],
    ignore_phase_start_s: float | None = None,
    ignore_phase_end_s: float | None = None,
    phase_threshold: float | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    ids = _body_ids(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, ids, 2] - command.robot_body_pos_w[:, ids, 2])
    in_phase = _in_reference_phase(command, ignore_phase_start_s, ignore_phase_end_s)
    if phase_threshold is None:
        return torch.any(error > threshold, dim=-1) & ~in_phase
    active_threshold = torch.where(in_phase, phase_threshold, threshold).unsqueeze(-1)
    return torch.any(error > active_threshold, dim=-1)


def reward_weight(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, reward_name: str, weight_stages: list[dict[str, float]]
) -> torch.Tensor:
    del env_ids
    cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > int(stage["step"]):
            cfg.weight = stage["weight"]
    return torch.tensor(cfg.weight, device=env.device)


# Re-export the small set of standard MDP terms used by the environment config.
generated_commands = base_mdp.generated_commands
base_lin_vel = base_mdp.base_lin_vel
base_ang_vel = base_mdp.base_ang_vel
joint_pos_rel = base_mdp.joint_pos_rel
joint_vel_rel = base_mdp.joint_vel_rel
last_action = base_mdp.last_action
JointPositionActionCfg = base_mdp.JointPositionActionCfg
reset_scene_to_default = base_mdp.reset_scene_to_default
action_rate_l2 = base_mdp.action_rate_l2
joint_pos_limits = base_mdp.joint_pos_limits
is_terminated = base_mdp.is_terminated
time_out = base_mdp.time_out
