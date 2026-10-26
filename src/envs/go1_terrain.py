"""Tiled terrains, independent curricula and ray height grids for batched Go1."""
from dataclasses import asdict,replace
import numpy as np
import torch
import mujoco

from .go1 import Go1WarpEnv
from .collisions import DEFAULT_COLLISION_GROUPS
from .terrain_config import TerrainConfig,TERRAIN_TYPES
from .terrains import build_terrain_model
from .height_sampler import TerrainHeightSampler
from .command_curriculum import RewardThresholdCurriculum
from .terrain_rewards import terrain_reward


class TerrainGo1Env(Go1WarpEnv):
    def __init__(self,cfg=None):
        cfg=cfg or TerrainConfig()
        if not isinstance(cfg,TerrainConfig): cfg=TerrainConfig.from_dict(asdict(cfg))
        cfg=replace(cfg,environment='terrain')
        cfg.validate_terrain()
        self.evaluation_mode=False
        super().__init__(cfg)
        print(f'[TerrainGo1Env] types={self.terrain_counts}; observations={self.num_observations}; '
              f'height_grid={cfg.height_grid_x}x{cfg.height_grid_y}; terrain_levels={cfg.terrain_levels}')

    def _load_model(self):
        model,self._origins_host=build_terrain_model(self.cfg)
        return model

    def _configure_model(self):
        # MJWarp's native BOX/MESH multicontact path requires zero margins.
        convex=np.isin(self.model.geom_type,[mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_MESH])
        self.model.geom_margin[convex]=0
        # Supplied task's pose, PD gains and 0.30 m spawn clearance.
        key=self.model.key('home').id
        self.model.key_qpos[key,2]=self.cfg.terrain_spawn_clearance
        self.model.key_qpos[key,7:]=[-.2,-.75,1.5,.2,.75,-1.5,.2,-1.,1.5,-.2,1.,-1.5]

    def _collision_groups(self):
        groups=dict(DEFAULT_COLLISION_GROUPS)
        groups['floor']=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i)
                         for i in range(self.model.ngeom)
                         if self.model.geom_group[i]==5 or self.model.geom(i).name=='floor']
        return groups

    def _extra_contact_groups(self):
        return ('trunk','FR_foot','FL_foot','RR_foot','RL_foot')

    def _initialize_domain(self):
        c=self.cfg
        self.num_height_rays=c.height_grid_x*c.height_grid_y
        self.num_observations=(61 if c.terrain_observation_layout=='legacy' else 48)+self.num_height_rays
        counts=[self.num_envs//len(c.terrain_types)+(i<self.num_envs%len(c.terrain_types))
                for i in range(len(c.terrain_types))]
        self.terrain_counts=dict(zip(c.terrain_types,counts))
        codes=[TERRAIN_TYPES.index(name) for name,count in zip(c.terrain_types,counts) for _ in range(count)]
        self.terrain_type=torch.tensor(codes,dtype=torch.long,device=self.device)
        self.terrain_level=torch.full((self.num_envs,),c.terrain_start_level,dtype=torch.long,device=self.device)
        self.terrain_completed=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.terrain_level[self.terrain_type==0]=0
        self.tile_origins=torch.tensor(self._origins_host,device=self.device)
        self.terrain_origin=self.tile_origins[self.terrain_type,self.terrain_level].clone()
        self.trunk_collision=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.foot_contact=torch.zeros((self.num_envs,4),dtype=torch.bool,device=self.device)
        self.command_tracking_sum=torch.zeros((self.num_envs,2),device=self.device)
        self.command_tracking_steps=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.command_bin=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.command_generator=torch.Generator(device=self.device).manual_seed(c.seed+1543)
        self.level_generator=torch.Generator(device=self.device).manual_seed(c.seed+4513)
        self.command_sampler=RewardThresholdCurriculum(self.num_envs,
                (c.command_vx,c.command_vy,c.command_wz),
                (c.command_bins_x,c.command_bins_y,c.command_bins_yaw),
                (c.command_initial_x,c.command_initial_y,c.command_initial_yaw),self.device,self.command_generator)
        self.height_sampler=TerrainHeightSampler(self)
        floor=self.model.geom('floor').id
        friction=float(self.model.geom_friction[floor,0])
        stiffness=float(-self.model.geom_solref[floor,0])
        damping=float(-self.model.geom_solref[floor,1])
        self.material_context=self.qpos.new_tensor([2*(friction-.4)/(1.5-.4)-1,
                    2*(stiffness-500)/(1500-500)-1,2*(damping-30)/(130-30)-1]).expand(self.num_envs,-1)
        self.mass_context=self.qpos.new_zeros((self.num_envs,1))
        self._terrain_initialized=False

    def _before_reset(self,ids,seed):
        c=self.cfg
        if seed is not None:
            self.command_generator.manual_seed(seed+1543)
            self.level_generator.manual_seed(seed+4513)
            self.height_sampler.generator.manual_seed(seed+8191)
        if self.evaluation_mode:
            # One deterministic level assignment per evaluation window, fixed
            # across within-window resets, so short episodes do not resample it.
            if seed is not None:
                levels=(torch.full((len(ids),),c.terrain_eval_level,device=self.device,dtype=torch.long)
                        if c.terrain_eval_level>=0 else torch.randint(c.terrain_levels,(len(ids),),
                                                generator=self.level_generator,device=self.device))
                self.terrain_level[ids]=torch.where(self.terrain_type[ids]==0,0,levels)
        elif self._terrain_initialized and c.terrain_curriculum:
            eligible=self.needs_reset[ids]&(self.terrain_type[ids]!=0)
            distance=(self.qpos[ids,:2]-self.terrain_origin[ids,:2]).norm(dim=-1)
            promote=distance>c.terrain_promotion_distance
            demote=(distance<c.terrain_demotion_factor*self.commands[ids,:2].norm(dim=-1)*
                    c.terrain_demotion_seconds)&~promote
            level=self.terrain_level[ids]+promote.long()-demote.long()
            completed=self.terrain_completed[ids]|(eligible&(level>=c.terrain_levels))
            random_level=torch.randint(c.terrain_levels,(len(ids),),generator=self.level_generator,device=self.device)
            level=torch.where(completed,random_level,level.clamp(0,c.terrain_levels-1))
            self.terrain_level[ids]=torch.where(eligible,level,self.terrain_level[ids])
            self.terrain_completed[ids]=completed
        self.terrain_origin[ids]=self.tile_origins[self.terrain_type[ids],self.terrain_level[ids]]
        self._terrain_initialized=True

    def _reset_pose(self,ids):
        self.qpos[ids,:2]=self.terrain_origin[ids,:2]
        self.qpos[ids,2]=self.terrain_origin[ids,2]+self.cfg.terrain_spawn_clearance
        self.foot_contact[ids]=False
        self.trunk_collision[ids]=False
        self.height_sampler.mark_reset(ids)

    def _after_physics(self):
        self.height_sampler.update()

    def _after_reset(self,ids):
        foot,_,_=self._terrain_contacts()
        self.foot_contact[ids]=foot[ids]

    def _sample_commands(self,ids):
        if ids.numel()==0: return
        c=self.cfg
        if c.command_curriculum and not self.evaluation_mode:
            steps=self.command_tracking_steps[ids].clamp_min(1)
            means=self.command_tracking_sum[ids]/steps[:,None]
            self.command_sampler.update(ids,self.command_bin[ids],means[:,0],means[:,1],
                        c.command_linear_threshold,c.command_angular_threshold,
                        c.command_local_range,c.command_weight_increment)
            commands,bins=self.command_sampler.sample(ids)
            self.command_bin[ids]=bins
        else:
            commands=(2*torch.rand((len(ids),3),device=self.device,generator=self.command_generator)-1)*self._command_limits
        commands[:,:2]*=(commands[:,:2].norm(dim=-1)>c.command_deadzone)[:,None]
        commands*=commands.abs()>c.command_deadzone
        self.commands[ids]=commands
        self.command_tracking_sum[ids]=0
        self.command_tracking_steps[ids]=0

    def _terrain_contacts(self):
        if self._reward_contacts is not None:
            _,flags=self._reward_contacts.update(self.warp_model,self._mjw,self.cfg.collision_force_threshold)
            force=self._reward_contacts.group_force_t
            foot=force[:,9:13,2]>self.cfg.terrain_foot_force_threshold
            trunk=force[:,8].norm(dim=-1)>self.cfg.terrain_trunk_force_threshold
            return foot,flags[:,:8],trunk
        contacts=self.collision_snapshot()
        legs=('FR','FL','RR','RL')
        groups=[f'{leg}_{part}' for part in ('calf','thigh') for leg in legs]
        groups+=['trunk']+[f'{leg}_foot' for leg in legs]
        force=self._collision_group_forces(contacts,groups)
        foot=force[:,9:13,2]>self.cfg.terrain_foot_force_threshold
        flags=force[:,:8].norm(dim=-1)>self.cfg.collision_force_threshold
        trunk=force[:,8].norm(dim=-1)>self.cfg.terrain_trunk_force_threshold
        return foot,flags,trunk

    def _reward(self,previous_actions,previous_joint_velocity):
        return terrain_reward(self,previous_actions,previous_joint_velocity)

    def step(self,actions):
        obs,reward,terminated,truncated,info=super().step(actions)
        info['terrain_type']=self.terrain_type.clone()
        info['terrain_level']=self.terrain_level.clone()
        return obs,reward,terminated,truncated,info

    def _termination(self,finite,gravity):
        center=(self.cfg.height_grid_x//2)*self.cfg.height_grid_y+self.cfg.height_grid_y//2
        ground=self.height_sampler.heights[:,center]
        terminated=(~finite)|self.trunk_collision|((self.qpos[:,2]-ground)<self.cfg.terrain_fall_height)
        distance=(self.qpos[:,:2]-self.terrain_origin[:,:2]).norm(dim=-1)
        truncated=(self.steps>=self.max_episode_steps)|(distance>=self.cfg.terrain_exit_radius)
        return terminated,truncated

    def observations(self):
        with self._scope(): self.height_sampler.refresh_pending()
        if self.cfg.terrain_observation_layout=='compact':
            state=super().observations()
        else:
            linear,angular,gravity=self.body_state()
            q=(self.qpos[:,self.q_index]-self.nominal)/4.6
            velocity=self.qvel[:,self.v_index]/35.
            state=torch.cat((q,velocity,self.actions/10.,self.foot_contact.float()*2-1,
                        (self.feet_air_time/2.5-1).clamp(-1,1),
                        (linear/10).clamp(-1,1),(angular/50).clamp(-1,1),self.commands,gravity,
                        (self.qpos[:,2:3]/self.cfg.nominal_height-1).clamp(-1,1),
                        self.mass_context,self.material_context),-1)
        return torch.cat((state,self.height_sampler.observation),-1)

    def get_curriculum_state(self):
        return {name:getattr(self,name).detach().cpu().clone() for name in
                ('terrain_type','terrain_level','terrain_completed')} | {
                'command_weights':self.command_sampler.weights.detach().cpu().clone(),
                'command_grid':self.command_sampler.grid.detach().cpu().clone()}

    def load_curriculum_state(self,state):
        if state['terrain_type'].shape!=self.terrain_type.shape or not torch.equal(
                state['terrain_type'].to(self.device),self.terrain_type):
            raise ValueError('Resuming terrain curricula requires the same world count and terrain-types')
        if (state['command_weights'].shape!=self.command_sampler.weights.shape or
                ('command_grid' in state and not torch.equal(state['command_grid'].to(self.device),
                                                              self.command_sampler.grid))):
            raise ValueError('Resuming command curricula requires the same bins and command limits')
        if bool(((state['terrain_level']<0)|(state['terrain_level']>=self.cfg.terrain_levels)).any()):
            raise ValueError('Saved terrain levels exceed the configured terrain-levels')
        self.terrain_level.copy_(state['terrain_level'])
        self.terrain_completed.copy_(state['terrain_completed'])
        self.command_sampler.weights.copy_(state['command_weights'])

    def render(self,world_id=0,throttle=True,full_state=False):
        if self.viewer is None:
            from .viewer import MujocoViewer
            self.viewer=MujocoViewer(self.model,self.dt,throttle=throttle)
            self.viewer.scene_option.geomgroup[5]=1
        self.viewer.scene_callback=self._draw_markers if self.cfg.height_markers else None
        return super().render(world_id,throttle,full_state)

    def _draw_markers(self,viewer):
        sampler=self.height_sampler
        world=viewer.world_id
        points=torch.cat((sampler.sample_xy[world],sampler.heights[world,:,None]),-1).cpu().numpy()
        valid=sampler.valid[world].cpu().numpy()
        for point,hit in zip(points,valid):
            if viewer.scene.ngeom>=viewer.scene.maxgeom: break
            point[2]+=.012 if hit else self.cfg.height_missing_value
            geom=viewer.scene.geoms[viewer.scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.012]*3),
                                point,np.eye(3).ravel(),np.array([0.,1.,0.,1.] if hit else [1.,0.,0.,1.]))
            viewer.scene.ngeom+=1
