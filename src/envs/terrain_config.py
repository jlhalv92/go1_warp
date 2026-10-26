"""Configuration for the supplied tiled rough-terrain task."""
from dataclasses import dataclass
import math

from .domain_randomized import DomainRandomizationConfig

TERRAIN_TYPES = ('flat', 'upstairs', 'downstairs', 'boxy')


@dataclass
class TerrainConfig(DomainRandomizationConfig):
    environment: str = 'terrain'
    kp: float = 20.
    kd: float = .5
    reset_joint_noise: float = 0.
    command_seconds: float = 10.
    terrain_types: tuple[str, ...] = TERRAIN_TYPES
    terrain_seed: int = 42
    terrain_levels: int = 8
    terrain_start_level: int = 0
    terrain_eval_level: int = -1  # -1: seeded uniform level per world, frozen within evaluation
    terrain_curriculum: bool = True
    terrain_tile_size: float = 7.
    terrain_boxy_tile_size: int = 10
    terrain_tile_spacing: float = 14.
    terrain_row_spacing: float = 14.
    terrain_num_stairs: int = 6
    terrain_step_depth: float = .24
    terrain_max_step_height: float = .15
    terrain_max_box_height: float = .30
    terrain_upstairs_buffer: float = 2.5
    terrain_downstairs_buffer: float = 2.
    terrain_base_thickness: float = .1
    terrain_background_z: float = -5.
    terrain_contact_timeconst: float = .01
    terrain_spawn_clearance: float = .30
    terrain_exit_radius: float = 5.
    terrain_promotion_distance: float = 2.5
    terrain_demotion_factor: float = .5
    terrain_demotion_seconds: float = 10.
    terrain_fall_height: float = .10
    terrain_trunk_force_threshold: float = 1.
    terrain_foot_force_threshold: float = 1.
    terrain_observation_layout: str = 'legacy'  # 61 + 187; compact: 48 + 187
    height_grid_x: int = 17
    height_grid_y: int = 11
    height_extent_x: float = .8
    height_extent_y: float = .5
    height_reference: float = .30
    height_ray_top: float = 3.
    height_max_distance: float = 10.
    height_noise: float = .02
    height_missing_value: float = .5
    height_clip: float = 0.  # zero disables clipping, matching active supplied code
    height_scale: float = 1.
    height_ray_bvh: bool = True
    height_markers: bool = True
    command_curriculum: bool = True
    command_bins_x: int = 21
    command_bins_y: int = 2
    command_bins_yaw: int = 21
    command_initial_x: float = 1.
    command_initial_y: float = 1.
    command_initial_yaw: float = 1.
    command_linear_threshold: float = .8
    command_angular_threshold: float = .5
    command_local_range: float = .5
    command_weight_increment: float = .2
    command_deadzone: float = .2
    reward_track_xy: float = 1.
    reward_track_yaw: float = .5
    reward_vertical: float = -2.
    reward_roll_pitch: float = -.05
    reward_tilt: float = 0.
    reward_joint_limits: float = -10.
    reward_joint_acceleration: float = -2.5e-7
    reward_joint_velocity: float = 0.
    reward_torque: float = -.0002
    reward_action_rate: float = -.01
    reward_collision: float = -1.
    reward_height: float = 0.
    reward_airtime: float = 1.
    reward_stand_still: float = 0.
    reward_xy_temperature: float = .25
    reward_yaw_temperature: float = .25

    def validate_terrain(self):
        if (not self.terrain_types or len(set(self.terrain_types)) != len(self.terrain_types)
                or not set(self.terrain_types) <= set(TERRAIN_TYPES)):
            raise ValueError(f'terrain-types must be a nonempty subset of {TERRAIN_TYPES}')
        if self.terrain_levels < 1 or not 0 <= self.terrain_start_level < self.terrain_levels:
            raise ValueError('terrain-start-level must be within terrain-levels')
        if not -1 <= self.terrain_eval_level < self.terrain_levels:
            raise ValueError('terrain-eval-level must be -1 or within terrain-levels')
        if self.terrain_observation_layout not in ('legacy', 'compact'):
            raise ValueError('Unknown terrain observation layout')
        if self.height_grid_x%2==0 or self.height_grid_y%2==0:
            raise ValueError('Height grid dimensions must be odd so the base origin is sampled')
        for name in ('terrain_tile_size', 'terrain_boxy_tile_size', 'terrain_tile_spacing',
                     'terrain_row_spacing', 'terrain_step_depth', 'terrain_base_thickness',
                     'terrain_spawn_clearance', 'terrain_exit_radius', 'height_grid_x',
                     'height_grid_y', 'height_max_distance', 'height_scale',
                     'reward_xy_temperature', 'reward_yaw_temperature'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('terrain_max_step_height','terrain_max_box_height', 'terrain_upstairs_buffer',
                     'terrain_downstairs_buffer', 'height_noise', 'height_clip',
                     'height_extent_x', 'height_extent_y', 'terrain_fall_height',
                     'terrain_foot_force_threshold', 'terrain_trunk_force_threshold',
                     'command_local_range', 'command_weight_increment', 'command_deadzone'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.terrain_num_stairs < 0 or self.terrain_num_stairs*self.terrain_step_depth >= self.terrain_tile_size/2:
            raise ValueError('Stairs must leave a central platform')
        if not all(math.isfinite(getattr(self,name)) for name in
                   ('height_ray_top','height_missing_value','height_reference','terrain_background_z')):
            raise ValueError('Ray and terrain reference heights must be finite')
        if self.terrain_background_z>=0:
            raise ValueError('terrain-background-z must be below tile surfaces to avoid overlapping ground')
        width = max(self.terrain_boxy_tile_size, self.terrain_tile_size+
                    2*max(self.terrain_upstairs_buffer,self.terrain_downstairs_buffer))
        if min(self.terrain_tile_spacing, self.terrain_row_spacing) <= width:
            raise ValueError('Tile/row spacing must exceed the largest tile including buffers')
        if self.terrain_contact_timeconst < 2*.005:
            raise ValueError('terrain-contact-timeconst must be >=0.01 for the bundled timestep')
        if self.height_ray_top <= max(self.terrain_num_stairs*self.terrain_max_step_height,
                                      self.terrain_max_box_height):
            raise ValueError('height-ray-top must exceed all terrain surfaces')
        if min(self.command_bins_x,self.command_bins_y,self.command_bins_yaw) < 2:
            raise ValueError('Command curriculum needs at least two bins per axis')
