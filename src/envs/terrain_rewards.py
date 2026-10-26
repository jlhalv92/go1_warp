"""Torch port of the supplied Reward, with the terrain task's coefficients."""
import torch


def terrain_reward(env, previous_actions, previous_joint_velocity):
    c=env.cfg
    linear,angular,_=env.body_state()
    q,velocity=env.qpos[:,env.q_index],env.qvel[:,env.v_index]
    w,x,y,z=env.qpos[:,3:7].unbind(-1)
    roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1))
    foot,flags,trunk=env._terrain_contacts()
    filtered=foot|env.last_foot_contact
    first=(env.feet_air_time>0)&filtered
    env.feet_air_time+=env.dt
    airtime=((env.feet_air_time-c.max_airtime)*first).sum(-1)
    # The supplied Airtime resets with the same filtered contact as touchdown.
    env.feet_air_time*=~filtered
    env.last_foot_contact.copy_(foot)
    env.foot_contact.copy_(foot)
    env.trunk_collision.copy_(trunk)
    track_xy=torch.exp(-(linear[:,:2]-env.commands[:,:2]).square().sum(-1)/c.reward_xy_temperature)
    track_yaw=torch.exp(-(angular[:,2]-env.commands[:,2]).square()/c.reward_yaw_temperature)
    terms={
        'track_xy': c.reward_track_xy*track_xy,
        'track_yaw': c.reward_track_yaw*track_yaw,
        'airtime': c.reward_airtime*airtime*(env.commands[:,:2].norm(dim=-1)>.1),
        'vertical': c.reward_vertical*linear[:,2].square(),
        'roll_pitch': c.reward_roll_pitch*angular[:,:2].square().sum(-1),
        'tilt': c.reward_tilt*(roll.square()+pitch.square()),
        'joint_limits': c.reward_joint_limits*((env.soft_joint_limits[:,0]-q).clamp_min(0)+
                                               (q-env.soft_joint_limits[:,1]).clamp_min(0)).sum(-1),
        'joint_acceleration': c.reward_joint_acceleration*((velocity-previous_joint_velocity)/env.dt).square().sum(-1),
        'joint_velocity': c.reward_joint_velocity*velocity.square().sum(-1),
        'torque': c.reward_torque*env.qfrc_actuator[:,env.v_index].square().sum(-1),
        'action_rate': c.reward_action_rate*(env.actions-previous_actions).square().sum(-1),
        'collision': c.reward_collision*flags.float().sum(-1),
        'height': c.reward_height*(env.qpos[:,2]-c.nominal_height).square(),
        'stand_still': c.reward_stand_still*(.02/env.dt)*(q-env.nominal).abs().sum(-1)*
                      (env.commands.norm(dim=-1)<.1),
    }
    env.command_tracking_sum+=torch.stack((track_xy,track_yaw),-1)
    env.command_tracking_steps+=1
    tracking=(1-(torch.cat((linear[:,:2],angular[:,2:3]),-1)-env.commands).abs()).mean(-1).clamp_min(0)
    return (sum(terms.values())*env.dt).clamp_min(0),terms,tracking
