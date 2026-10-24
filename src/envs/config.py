"""Simulation configuration shared by the launchers and environment variants."""
from dataclasses import dataclass, fields


@dataclass
class EnvConfig:
    environment: str = 'flat'
    num_envs: int = 1024
    device: str = 'cuda:0'
    backend: str = 'warp'  # 'mujoco' is a slow reference backend for debugging
    xml_path: str | None = None
    collision_only: bool = False  # Legacy loader flag: strips visual meshes only
    decimation: int = 4  # XML dt=0.005 -> control dt=0.02 s
    episode_seconds: float = 20.0
    kp: float = 30.0
    kd: float = 0.8
    action_scale: float = 0.25
    nconmax: int = 128
    njmax: int = 512
    seed: int = 0
    reset_joint_noise: float = 0.03
    command_seconds: float = 5.0
    command_vx: float = 1.0
    command_vy: float = 0.5
    command_wz: float = 1.0
    nominal_height: float = 0.34
    soft_joint_position_limit: float = 0.9
    use_physics_graph: bool = True  # Controls step, forward and reset graphs
    solver_iterations: int = 100
    solver_ls_iterations: int = 50
    debug_checks: bool = False
    max_airtime: float = 0.5
    collision_force_threshold: float = 0.1
    fast_contacts: bool = True

    @classmethod
    def from_dict(cls, values):
        """Load saved configuration, dropping fields retired by this variant."""
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})



# Compatibility name for callers that previously imported Config.
Config = EnvConfig
