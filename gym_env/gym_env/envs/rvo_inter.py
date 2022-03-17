from gym_env.envs.reciprocal_vel_obs import reciprocal_vel_obs
from math import sqrt, atan2, asin, sin, pi, cos, inf
import numpy as np

class rvo_inter(reciprocal_vel_obs):
    def __init__(self, neighbor_region=5, neighbor_num=10, vxmax=1.5, vymax=1.5, acceler=0.5, env_train=True, exp_radius=0.2, ctime_threshold=5, ctime_line_threshold=1):
        super(rvo_inter, self).__init__(neighbor_region, vxmax, vymax, acceler)

        self.env_train = env_train
        self.exp_radius = exp_radius
        self.nm = neighbor_num
        self.ctime_threshold = ctime_threshold
        self.ctime_line_threshold = ctime_line_threshold

    def config_vo_inf(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action=np.zeros((2,)), **kwargs):
        # mode: vo, rvo, hrvo
        robot_state, ns_list, oc_list, ol_list = self.preprocess(robot_state, nei_state_list, obs_cir_list, obs_line_list)

        action = np.squeeze(action)

        vo_list1 = list(map(lambda x: self.config_vo_circle2(robot_state, x, action, 'rvo', **kwargs), ns_list))
        vo_list2 = list(map(lambda y: self.config_vo_circle2(robot_state, y,action, 'vo', **kwargs), oc_list))
        vo_list3 = list(map(lambda z: self.config_vo_line2(robot_state, z, action, **kwargs), ol_list))

        obs_vo_list = []

        collision_flag = False
        vo_flag = False
        min_exp_time = inf

        for vo_inf in vo_list1 + vo_list2 + vo_list3:
            obs_vo_list.append(vo_inf[0])
            if vo_inf[1] is True:
                # obs_vo_list.append(vo_inf[0])
                vo_flag = True
                if vo_inf[2] < min_exp_time:
                    min_exp_time = vo_inf[2]
            
            if vo_inf[3] is True: collision_flag = True

        obs_vo_list.sort(reverse=True, key=lambda x: (-x[-1], x[-2]))

        if len(obs_vo_list) > self.nm:
            obs_vo_list_nm = obs_vo_list[-self.nm:]
        else:
            obs_vo_list_nm = obs_vo_list

        if self.nm == 0:
            obs_vo_list_nm = []

        return obs_vo_list_nm, vo_flag, min_exp_time, collision_flag
        
    def config_vo_reward(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action=np.zeros((2,)), **kwargs):

        robot_state, ns_list, oc_list, ol_list = self.preprocess(robot_state, nei_state_list, obs_cir_list, obs_line_list)

        vo_list1 = list(map(lambda x: self.config_vo_circle2(robot_state, x, action, 'rvo', **kwargs), ns_list))
        vo_list2 = list(map(lambda y: self.config_vo_circle2(robot_state, y,action, 'vo', **kwargs), oc_list))
        vo_list3 = list(map(lambda z: self.config_vo_line2(robot_state, z, action, **kwargs), ol_list))

        vo_flag = False
        min_exp_time = inf
        min_dis = inf

        for vo_inf in vo_list1 + vo_list2 + vo_list3:

            if vo_inf[4] < min_dis:
                min_dis = vo_inf[4]

            if vo_inf[1] is True:
                vo_flag = True
                if vo_inf[2] < min_exp_time:
                    min_exp_time = vo_inf[2]

        new_des_list = [des_v[-1] for des_v in vo_list3]
        
        return vo_flag, min_exp_time, min_dis


    def config_vo_observe(self, robot_state, nei_state_list, obs_cir_list, obs_line_list):

        vo_list, _, _, _ = self.config_vo_inf(robot_state, nei_state_list, obs_cir_list, obs_line_list)
        
        return vo_list

    def config_vo_circle2(self, state, circular, action, mode='rvo', **kwargs):
        
        x, y, vx, vy, r = state[0:5]
        mx, my, mvx, mvy, mr = circular[0:5]

        if mvx == 0 and mvy == 0:
            mode = 'vo'

        vo_flag = False
        collision_flag = False

        rel_x = x - mx
        rel_y = y - my

        dis_mr = sqrt((rel_y)**2 + (rel_x)**2)
        angle_mr = atan2(my - y, mx - x)

        real_dis_mr = sqrt((rel_y)**2 + (rel_x)**2)
        
        env_train = kwargs.get('env_train', self.env_train)

        if env_train:
            if dis_mr <= r + mr:
                dis_mr = r + mr
                collision_flag = True
        else:
            if dis_mr <= r - self.exp_radius + mr:
                collision_flag = True

            if dis_mr <= r + mr:
                dis_mr = r + mr
                
        ratio = (r + mr)/dis_mr
        half_angle = asin( ratio ) 
        line_left_ori = reciprocal_vel_obs.wraptopi(angle_mr + half_angle) 
        line_right_ori = reciprocal_vel_obs.wraptopi(angle_mr - half_angle) 

        if mode == 'vo':
            vo = [mvx, mvy, line_left_ori, line_right_ori]
            rel_vx = action[0] - mvx 
            rel_vy = action[1] - mvy

        elif mode == 'rvo':
            vo = [(vx + mvx)/2, (vy + mvy)/2, line_left_ori, line_right_ori] 
            rel_vx = 2*action[0] - mvx - vx
            rel_vy = 2*action[1] - mvy - vy

        exp_time = inf

        if self.vo_out_jud_vector(action[0], action[1], vo):
            vo_flag = False
            exp_time = inf
        else:
            exp_time = self.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, r + mr)
            if exp_time < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                exp_time = inf
            
        input_exp_time = 1 / (exp_time+0.2)
        min_dis = real_dis_mr-mr

        observation_vo = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), min_dis, input_exp_time]
        
        return [observation_vo, vo_flag, exp_time, collision_flag, min_dis]

    def config_vo_line2(self, robot_state, line, action, **kwargs):

        x, y, vx, vy, r = robot_state[0:5]
    
        apex = [0, 0]

        theta1 = atan2(line[0][1] - y, line[0][0] - x)
        theta2 = atan2(line[1][1] - y, line[1][0] - x)
        
        dis_mr1 = sqrt( (line[0][1] - y)**2 + (line[0][0] - x)**2 )
        dis_mr2 = sqrt( (line[1][1] - y)**2 + (line[1][0] - x)**2 )

        half_angle1 = asin(reciprocal_vel_obs.clamp(r/dis_mr1, 0, 1))
        half_angle2 = asin(reciprocal_vel_obs.clamp(r/dis_mr2, 0, 1))
        
        if reciprocal_vel_obs.wraptopi(theta2-theta1) > 0:
            line_left_ori = reciprocal_vel_obs.wraptopi(theta2 + half_angle2)
            line_right_ori = reciprocal_vel_obs.wraptopi(theta1 - half_angle1)
        else:
            line_left_ori = reciprocal_vel_obs.wraptopi(theta1 + half_angle1)
            line_right_ori = reciprocal_vel_obs.wraptopi(theta2 - half_angle2)

        vo = apex + [line_left_ori, line_right_ori]
        exp_time = inf

        temp = robot_state[0:2]
        temp1 = line

        p2s, p2s_angle = rvo_inter.point2segment(temp, temp1)

        env_train = kwargs.get('env_train', self.env_train)

        if env_train:
            collision_flag = True if p2s <= r else False
        else:
            collision_flag = True if p2s <= r - self.exp_radius else False

        if self.vo_out_jud_vector(action[0], action[1], vo):
            vo_flag = False
        else:
            exp_time = reciprocal_vel_obs.exp_collision_segment(line, x, y, action[0], action[1], r)
            if exp_time < self.ctime_line_threshold:
                vo_flag = True
            else:
                vo_flag = False
                exp_time = inf

        input_exp_time = 1 / (exp_time+0.2)
        min_dis = p2s

        observation_vo = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), min_dis, input_exp_time]
        
        return [observation_vo, vo_flag, exp_time, collision_flag, min_dis]

    def vo_out_jud(self, vx, vy, vo):
        
        theta = atan2(vy - vo[1], vx - vo[0])

        if reciprocal_vel_obs.between_angle(vo[2], vo[3], theta):
            return False
        else:
            return True

    def vo_out_jud_vector(self, vx, vy, vo):
        
        rel_vector = [vx - vo[0], vy - vo[1]]
        line_left_vector = [cos(vo[2]), sin(vo[2]) ]
        line_right_vector = [cos(vo[3]), sin(vo[3]) ]
        
        if reciprocal_vel_obs.between_vector(line_left_vector, line_right_vector, rel_vector):
            return False
        else:
            return True

    @staticmethod
    def point2segment(point, segment):
        
        point = np.squeeze(point[0:2])
        sp = segment[0]
        ep = segment[1]

        l2 = (ep - sp) @ (ep - sp)

        if (l2 == 0.0):
            return np.linalg.norm(point - sp), 

        t = max(0, min(1, ((point-sp) @ (ep-sp)) / l2 ))

        projection = sp + t * (ep-sp)

        relative = projection - point

        distance = np.linalg.norm(relative) 
        angle = atan2( relative[1], relative[0] )

        return distance, angle