# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_assets import G1_29DOF_CFG
from isaaclab_physx.physics import PhysxCfg

from . import mdp

REPO_ROOT = Path(__file__).resolve().parents[4]
MOTION_FILE = REPO_ROOT / "assets" / "motions" / "g1" / "halay_loop.npz"


def ground_slope_quat() -> tuple[float, float, float, float]:
    """Return an xyzw quaternion for the requested platform pitch."""
    slope_deg = float(os.environ.get("G1_GROUND_SLOPE_DEG", "0.0"))
    if abs(slope_deg) > 30.0:
        raise ValueError("G1_GROUND_SLOPE_DEG must be between -30 and 30 degrees.")
    half_angle = 0.5 * math.radians(slope_deg)
    return (0.0, math.sin(half_angle), 0.0, math.cos(half_angle))

ACTION_SCALE = {
    ".*_elbow_joint": 0.43857731392336724,
    ".*_shoulder_pitch_joint": 0.43857731392336724,
    ".*_shoulder_roll_joint": 0.43857731392336724,
    ".*_shoulder_yaw_joint": 0.43857731392336724,
    ".*_wrist_roll_joint": 0.43857731392336724,
    ".*_hip_pitch_joint": 0.5475464629911068,
    ".*_hip_yaw_joint": 0.5475464629911068,
    "waist_yaw_joint": 0.5475464629911068,
    ".*_hip_roll_joint": 0.35066146637882434,
    ".*_knee_joint": 0.35066146637882434,
    ".*_wrist_pitch_joint": 0.07450087032950714,
    ".*_wrist_yaw_joint": 0.07450087032950714,
    "waist_pitch_joint": 0.43857731392336724,
    "waist_roll_joint": 0.43857731392336724,
    ".*_ankle_pitch_joint": 0.43857731392336724,
    ".*_ankle_roll_joint": 0.43857731392336724,
}


def make_robot_cfg() -> ArticulationCfg:
    """Return the Isaac Lab G1 asset with MJLab-compatible defaults and PD gains."""
    cfg = G1_29DOF_CFG.copy()
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.init_state.pos = (0.0, 0.0, 0.76)
    cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    cfg.init_state.joint_pos = {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }
    hands = cfg.actuators["hands"]
    cfg.actuators = {
        "shoulders_elbows_wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_elbow_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_wrist_roll_joint",
            ],
            joint_effort_limit=25.0,
            stiffness=14.25062309787429,
            damping=0.907222843292423,
            armature=0.003609725,
        ),
        "hip_pitch_yaw_waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"],
            joint_effort_limit=88.0,
            stiffness=40.17923863450712,
            damping=2.557889775413375,
            armature=0.01017752004132231,
        ),
        "hip_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
            joint_effort_limit=139.0,
            stiffness=99.09842777666111,
            damping=6.308801853496639,
            armature=0.025101925,
        ),
        "wrist_pitch_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            joint_effort_limit=5.0,
            stiffness=16.77832748089279,
            damping=1.06814150219,
            armature=0.00425,
        ),
        "waist_ankles": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch_joint", "waist_roll_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            joint_effort_limit=50.0,
            stiffness=28.50124619574858,
            damping=1.814445686584846,
            armature=0.00721945,
        ),
        "hands": hands,
    }
    return cfg


@configclass
class G1TrackingSceneCfg(InteractiveSceneCfg):
    """Flat scene containing the 29 controlled G1 joints."""

    # A collision-enabled kinematic platform is cloned under every environment.
    # Unlike the visual mesh of GroundPlaneCfg, rotating this prim changes the
    # actual PhysX contact surface. Set its launch angle with
    # G1_GROUND_SLOPE_DEG, or manipulate one env's Ground prim in Kit.
    ground = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(2.4, 2.4, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.20, 0.24),
                roughness=0.8,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.05),
            rot=ground_slope_quat(),
        ),
    )
    robot: ArticulationCfg = make_robot_cfg()
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0)
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(mdp.G1_JOINT_NAMES),
        preserve_order=True,
        scale=ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        # Draw the NPZ target as a cyan ghost skeleton in only the environment
        # nearest the center of the tiled scene. This is visualization-only.
        debug_vis=True,
        motion_file=str(MOTION_FILE),
        anchor_body_name="torso_link",
        body_names=mdp.TRACKED_BODY_NAMES,
        joint_names=mdp.G1_JOINT_NAMES,
        start_time_s=4.18,
        pose_range={
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.005, 0.005),
            "roll": (-0.03, 0.03),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.05, 0.05),
        },
        velocity_range={
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.05, 0.05),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        joint_position_range=(-0.03, 0.03),
    )


JOINT_CFG = SceneEntityCfg("robot", joint_names=list(mdp.G1_JOINT_NAMES), preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class ActorCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": JOINT_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": JOINT_CFG},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": JOINT_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    actor: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()


JUMP_PHASE = {"phase_start_s": 59 / 50, "phase_end_s": 101 / 50, "phase_scale": 0.5}


@configclass
class RewardsCfg:
    motion_global_root_pos = RewTerm(
        func=mdp.motion_anchor_position_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3, **JUMP_PHASE},
    )
    motion_global_root_ori = RewTerm(
        func=mdp.motion_anchor_orientation_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4, **JUMP_PHASE},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_body_position_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3, **JUMP_PHASE},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_body_orientation_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4, **JUMP_PHASE},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_body_linear_velocity_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0, **JUMP_PHASE},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_body_angular_velocity_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14, **JUMP_PHASE},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": JOINT_CFG})
    body_orientation_l2 = RewTerm(
        func=mdp.torso_orientation_l2,
        weight=-12.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    body_ang_vel = RewTerm(
        func=mdp.torso_angular_velocity_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_height,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "phase_threshold": 0.50,
            "phase_start_s": JUMP_PHASE["phase_start_s"],
            "phase_end_s": JUMP_PHASE["phase_end_s"],
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_orientation, params={"command_name": "motion", "threshold": 0.8}
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_end_effector_height,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
            "ignore_phase_start_s": JUMP_PHASE["phase_start_s"],
            "ignore_phase_end_s": JUMP_PHASE["phase_end_s"],
            "phase_threshold": 0.50,
        },
    )


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class CurriculumCfg:
    body_orientation_weight = CurrTerm(
        func=mdp.reward_weight,
        params={
            "reward_name": "body_orientation_l2",
            "weight_stages": [
                {"step": 15_000 * 24, "weight": -4.0},
                {"step": 25_000 * 24, "weight": -1.0},
            ],
        },
    )
    body_ang_vel_weight = CurrTerm(
        func=mdp.reward_weight,
        params={
            "reward_name": "body_ang_vel",
            "weight_stages": [
                {"step": 15_000 * 24, "weight": -0.08},
                {"step": 25_000 * 24, "weight": -0.03},
            ],
        },
    )


@configclass
class G1TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Isaac Lab equivalent of Unitree-G1-Tracking-No-State-Estimation."""

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        # Isaac Sim PhysX is intentional: this task is meant to run in the Kit
        # viewport with many tiled G1 environments visible at the same time.
        physics=PhysxCfg(),
    )
    scene: G1TrackingSceneCfg = G1TrackingSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 10.0

    def play_mode(self) -> None:
        super().play_mode()
        self.episode_length_s = 1.0e9
        self.observations.actor.enable_corruption = False
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.curriculum = {}
