"""Yaw-aligned terrain rays, with cached outputs and a MuJoCo CPU reference.

All worlds share the same static atlas. GPU rays therefore query one immutable
reference scene, flattening (world, ray) into its ray axis. The robot is excluded
by geom groups. This avoids a separate copy of the ray BVH for every world.
"""
import numpy as np
import torch
import mujoco


class TerrainHeightSampler:
    def __init__(self, env):
        self.env, self.cfg = env, env.cfg
        c=self.cfg
        x=torch.linspace(-c.height_extent_x,c.height_extent_x,c.height_grid_x,device=env.device)
        y=torch.linspace(-c.height_extent_y,c.height_extent_y,c.height_grid_y,device=env.device)
        if c.height_grid_x==1: x.zero_()
        if c.height_grid_y==1: y.zero_()
        self.local_xy=torch.stack(torch.meshgrid(x,y,indexing='ij'),-1).reshape(-1,2)
        self.num_rays=self.local_xy.shape[0]
        shape=(env.num_envs,self.num_rays)
        self.heights=torch.zeros(shape,device=env.device)
        self.valid=torch.zeros(shape,dtype=torch.bool,device=env.device)
        self.observation=torch.zeros(shape,device=env.device)
        self.sample_xy=torch.zeros((*shape,2),device=env.device)
        self.geom_ids=torch.full(shape,-1,dtype=torch.int32,device=env.device)
        self.generator=torch.Generator(device=env.device).manual_seed(c.seed+8191)
        self.geomgroup=np.array([1,0,0,0,0,1],dtype=np.uint8)
        self._all_ids=torch.arange(env.num_envs,device=env.device)
        self._pending=None
        self._views={}
        self.ray_context=None
        if env._wp is not None:
            wp,mjw=env._wp,env._mjw
            count=env.num_envs*self.num_rays
            self.origins_wp=wp.zeros((1,count),dtype=wp.vec3,device=str(env.device))
            self.directions_wp=wp.zeros((1,count),dtype=wp.vec3,device=str(env.device))
            self.dist_wp=wp.empty((1,count),dtype=float,device=str(env.device))
            self.ids_wp=wp.empty((1,count),dtype=int,device=str(env.device))
            self.normal_wp=wp.empty((1,count),dtype=wp.vec3,device=str(env.device))
            self.exclude_wp=wp.full(count,-1,dtype=int,device=str(env.device))
            self.origins=wp.to_torch(self.origins_wp).view(-1,3)
            self.directions=wp.to_torch(self.directions_wp).view(-1,3)
            self.directions[:,2]=-1
            self.dist=wp.to_torch(self.dist_wp).view(-1)
            self.hit_ids=wp.to_torch(self.ids_wp).view(-1)
            self.normals=wp.to_torch(self.normal_wp).view(-1,3)
            self.exclude=wp.to_torch(self.exclude_wp)
            reference=mujoco.MjData(env.model)
            mujoco.mj_forward(env.model,reference)
            self.reference_data=mjw.put_data(env.model,reference,nworld=1,
                                             nconmax=max(128,reference.ncon),njmax=max(512,reference.nefc))
            if c.height_ray_bvh:
                self.ray_context=mjw.create_render_context(env.model,nworld=1,cam_active=[],
                                                           enabled_geom_groups=[0,5],use_textures=False)

    def mark_reset(self, ids):
        self._pending=ids

    def refresh_pending(self):
        if self._pending is not None:
            ids,self._pending=self._pending,None
            self.update(ids)

    def _buffers(self, count):
        if count not in self._views:
            wp=self.env._wp
            self._views[count]=(
                wp.from_torch(self.origins[:count].view(1,count,3),dtype=wp.vec3),
                wp.from_torch(self.directions[:count].view(1,count,3),dtype=wp.vec3),
                wp.from_torch(self.dist[:count].view(1,count)),
                wp.from_torch(self.hit_ids[:count].view(1,count)),
                wp.from_torch(self.normals[:count].view(1,count,3),dtype=wp.vec3))
        return self._views[count]

    def update(self, ids=None):
        ids=self._all_ids if ids is None else ids
        if ids.numel()==0: return
        q=self.env.qpos[ids]
        pose_valid=torch.isfinite(q).all(-1)
        q=torch.where(pose_valid[:,None],q,self.env.home[None])
        w,x,y,z=q[:,3:7].unbind(-1)
        yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        c,s=yaw.cos()[:,None],yaw.sin()[:,None]
        xy=torch.stack((q[:,None,0]+c*self.local_xy[:,0]-s*self.local_xy[:,1],
                        q[:,None,1]+s*self.local_xy[:,0]+c*self.local_xy[:,1]),-1)
        self.sample_xy[ids]=xy
        worlds=ids.numel()
        if self.env._wp is not None:
            count=worlds*self.num_rays
            self.origins[:count,:2]=xy.reshape(-1,2)
            self.origins[:count,2]=self.cfg.height_ray_top
            origins,directions,dist,geomid,normal=self._buffers(count)
            # Static reference scene: no physics state copy, refit or host count.
            self.env._mjw.rays(self.env.warp_model,self.reference_data,origins,directions,
                               tuple(int(x) for x in self.geomgroup),True,
                               self._body_exclude(count),dist,geomid,normal,rc=self.ray_context)
            distance=self.dist[:count].view(worlds,self.num_rays)
            hit=self.hit_ids[:count].view(worlds,self.num_rays)
        else:
            distance=torch.full((worlds,self.num_rays),-1.)
            hit=torch.full_like(distance,-1,dtype=torch.int32)
            origin=np.array([0.,0.,self.cfg.height_ray_top])
            direction=np.array([0.,0.,-1.]); hit_id=np.empty(1,dtype=np.int32)
            for row,world in enumerate(ids.tolist()):
                if not pose_valid[row]: continue
                for ray,point in enumerate(xy[row].numpy()):
                    origin[:2]=point
                    value=mujoco.mj_ray(self.env.model,self.env.cpu_data[world],origin,direction,
                                        self.geomgroup,True,-1,hit_id)
                    distance[row,ray]=value; hit[row,ray]=int(hit_id[0])
        valid=(distance>=0)&(distance<=self.cfg.height_max_distance)&pose_valid[:,None]
        fallback=q[:,2,None]-self.cfg.height_reference-self.cfg.height_missing_value
        heights=torch.where(valid,self.cfg.height_ray_top-distance,fallback)
        relative=q[:,2,None]-heights-self.cfg.height_reference
        if self.cfg.height_noise:
            relative=relative+(2*torch.rand(relative.shape,generator=self.generator,
                                            device=self.env.device)-1)*self.cfg.height_noise
        if self.cfg.height_clip:
            relative=relative.clamp(-self.cfg.height_clip,self.cfg.height_clip)
        self.heights[ids]=heights
        self.valid[ids]=valid
        self.geom_ids[ids]=torch.where(valid,hit,-1)
        self.observation[ids]=relative*self.cfg.height_scale

    def _body_exclude(self,count):
        key=('exclude',count)
        if key not in self._views:
            self._views[key]=self.env._wp.from_torch(self.exclude[:count])
        return self._views[key]
