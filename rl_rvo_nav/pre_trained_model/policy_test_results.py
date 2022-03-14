import gym
import gym_env
import pickle
import numpy as np
import gym
import sys
from rl_rvo_nav.policy_test.post_train import post_train
from pathlib import Path
# from crowd_nav.utils.explorer import Explorer
# from crowd_nav.policy.policy_factory import policy_factory
# from crowd_sim.envs.utils.robot import Robot
# from crowd_sim.envs.policy.orca import ORCA

counter = 2
robot_num = 6

save_name = 'r10_{}'
world_path = 'policy_test_world.yaml'
# world_path = 'policy_test_world_lines.yaml'

abs_path = Path(sys.path[0]) 
save_path_string = str( abs_path/ 'pre_trained' /save_name)

# fname_model = save_path_string +'_1000.pt'
fname_model = save_path_string + '_check_point_1000.pt'
arg_model = save_path_string

figure_save_path = Path(save_path_string.format(counter, counter) + 'figure')
ani_save_path = Path(save_path_string.format(counter, counter) + 'ani')

r = open(arg_model.format(counter, counter), 'rb')
args = pickle.load(r) 

neighbors_num = 5

env = gym.make('mrnav-v0', world_name=world_path, robot_number=robot_num, neighbors_region=4, neighbors_num=neighbors_num, robot_init_mode=3, random_bear=True, reward_parameter=args.reward_parameter, obs_mode=0, reward_mode=args.reward_mode, full=False, env_train=False)

pt = post_train(env, num_episodes=100, reset_mode=3, render=True, std_factor=0.001, neighbors_num=neighbors_num, save=False, max_ep_len=300, acceler_vel=1, args=args)
pt.policy_test('drl', fname_model.format(counter, counter), counter, result_path=str(abs_path), result_name='/result.txt', figure_save_path=figure_save_path)

# env.env_gym.world_plot.create_animate(figure_save_path, ani_save_path, ani_name='test_drl_random_'+str(robot_num), keep_len=30, rm_fig_path=True)
# pt.policy_test('orca', lstm_rl_model, counter, result_path=str(cur_path), result_name='/result.txt', figure_save_path=figure_save_path)
