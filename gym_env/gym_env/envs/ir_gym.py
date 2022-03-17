from ir_sim.env import env_base
from math import sqrt, pi
from gym import spaces
from gym_env.envs.rvo_inter import rvo_inter
import numpy as np

class ir_gym(env_base):
    def __init__(self, world_name, neighbors_region=5, neighbors_num=10, vxmax = 1.5, vymax = 1.5, env_train=True, acceler = 0.5, **kwargs):
        super(ir_gym, self).__init__(world_name=world_name, **kwargs)

        # self.obs_mode = kwargs.get('obs_mode', 0)    # 0 drl_rvo, 1 drl_nrvo
        # self.reward_mode = kwargs.get('reward_mode', 0)

        self.radius_exp = kwargs.get('radius_exp', 0.2)

        self.env_train = env_train

        self.nr = neighbors_region
        self.nm = neighbors_num

        self.rvo = rvo_inter(neighbors_region, neighbors_num, vxmax, vymax, acceler, env_train, self.radius_exp)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        self.reward_parameter = kwargs.get('reward_parameter', (0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 20)) 
        self.acceler = acceler
        self.arrive_flag_cur = False

        self.rvo_state_dim = 8
        

    def cal_des_omni_list(self):
        des_vel_list = [robot.cal_des_vel_omni() for robot in self.robot_list]
        return des_vel_list


    def rvo_reward_list_cal(self, action_list, **kwargs):    
        ts = self.components['robots'].total_states() # robot_state_list, nei_state_list, obs_circular_list, obs_line_list

        rvo_reward_list = list(map(lambda robot_state, action: self.rvo_reward_cal(robot_state, ts[1], ts[2], ts[3], action, self.reward_parameter, **kwargs), ts[0], action_list))

        return rvo_reward_list
    
    def rvo_reward_cal(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20), **kwargs):
        
        vo_flag, min_exp_time, min_dis = self.rvo.config_vo_reward(robot_state, nei_state_list, obs_cir_list, obs_line_list, action, **kwargs)

        des_vel = np.round(np.squeeze(robot_state[-2:]), 2)
        
        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter

        dis_des = sqrt((action[0] - des_vel[0] )**2 + (action[1] - des_vel[1])**2)
        max_dis_des = 3
        dis_des_reward = - dis_des / max_dis_des #  (0-1)
        exp_time_reward = - 0.2/(min_exp_time+0.2) # (0-1)
        
        # rvo reward    
        if vo_flag:
            rvo_reward = p2 + p3 * dis_des_reward + p4 * exp_time_reward
            
            if min_exp_time < 0.1:
                rvo_reward = p2 + p1 * p4 * exp_time_reward
        else:
            rvo_reward = p5 + p6 * dis_des_reward
        
        rvo_reward = np.round(rvo_reward, 2)

        return rvo_reward

    def obs_move_reward_list(self, action_list, **kwargs):
        ts = self.components['robots'].total_states() # robot_state_list, nei_state_list, obs_circular_list, obs_line_list

        obs_reward_list = list(map(lambda robot, action: self.observation_reward(robot, ts[1], ts[2], ts[3], action, **kwargs), self.robot_list, action_list))

        obs_list = [l[0] for l in obs_reward_list]
        reward_list = [l[1] for l in obs_reward_list]
        done_list = [l[2] for l in obs_reward_list]
        info_list = [l[3] for l in obs_reward_list]

        return obs_list, reward_list, done_list, info_list

    def observation_reward(self, robot, nei_state_list, obs_circular_list, obs_line_list, action, **kwargs):

        robot_omni_state = robot.omni_state()
        des_vel = np.squeeze(robot.cal_des_vel_omni())
       
        done = False

        if robot.arrive() and not robot.arrive_flag:
            robot.arrive_flag = True
            arrive_reward_flag = True
        else:
            arrive_reward_flag = False

        obs_vo_list, vo_flag, min_exp_time, collision_flag = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, obs_circular_list, obs_line_list, action, **kwargs)

        radian = robot.state[2]
        cur_vel = np.squeeze(robot.vel_omni)
        radius = robot.radius_collision* np.ones(1,)

        propri_obs = np.concatenate([ cur_vel, des_vel, radian, radius]) 
        
        if len(obs_vo_list) == 0:
            exter_obs = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs = np.concatenate(obs_vo_list) # vo list
            
        observation = np.round(np.concatenate([propri_obs, exter_obs]), 2)

        # dis2goal = sqrt( robot.state[0:2] - robot.goal[0:2])
        mov_reward = self.mov_reward(collision_flag, arrive_reward_flag, self.reward_parameter, min_exp_time)

        reward = mov_reward

        done = True if collision_flag else False
        info = True if robot.arrive_flag else False
        
        return [observation, reward, done, info]

    def mov_reward(self, collision_flag, arrive_reward_flag, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 15), min_exp_time=100, dis2goal=100):

        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter

        collision_reward = p7 if collision_flag else 0
        arrive_reward = p8 if arrive_reward_flag else 0
        time_reward = 0
        
        mov_reward = collision_reward + arrive_reward + time_reward

        return mov_reward

    def osc_reward(self, state_list):
        # to avoid oscillation
        dif_rad_list = []
        
        if len(state_list) < 3:
            return 0

        for i in range(len(state_list) - 1):
            dif = ir_gym.wraptopi(state_list[i+1][2, 0] - state_list[i][2, 0])
            dif_rad_list.append(round(dif, 2))

        for j in range(len(dif_rad_list)-3):
            
            if dif_rad_list[j] * dif_rad_list[j+1] < -0.05 and dif_rad_list[j+1] * dif_rad_list[j+2] < -0.05 and dif_rad_list[j+2] * dif_rad_list[j+3] < -0.05:
                print('osc', dif_rad_list[j], dif_rad_list[j+1], dif_rad_list[j+2], dif_rad_list[j+3])
                return -10
        return 0

    def observation(self, robot, nei_state_list, obs_circular_list, obs_line_list):

        robot_omni_state = robot.omni_state()
        des_vel = np.squeeze(robot_omni_state[-2:])
        
        obs_vo_list, _, min_exp_time, _ = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, obs_circular_list, obs_line_list)
    
        cur_vel = np.squeeze(robot.vel_omni)
        radian = robot.state[2]
        radius = robot.radius_collision* np.ones(1,)

        if len(obs_vo_list) == 0:
            exter_obs = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs = np.concatenate(obs_vo_list) # vo list

        
        propri_obs = np.concatenate([ cur_vel, des_vel, radian, radius]) 
        observation = np.round(np.concatenate([propri_obs, exter_obs]), 2)

        return observation

    def env_reset(self, reset_mode=1, **kwargs):
        
        self.components['robots'].robots_reset(reset_mode, **kwargs)
        ts = self.components['robots'].total_states()
        obs_list = list(map(lambda robot: self.observation(robot, ts[1], ts[2], ts[3]), self.robot_list))

        return obs_list

    def env_reset_one(self, id):
        self.robot_reset(id)

    def env_observation(self):
        ts = self.components['robots'].total_states()
        obs_list = list(map(lambda robot: self.observation(robot, ts[1], ts[2], ts[3]), self.robot_list))

        return obs_list

    @staticmethod
    def wraptopi(theta):

        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta
    