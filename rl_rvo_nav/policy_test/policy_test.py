import gym
import gym_env
from pathlib import Path
import pickle
import sys
from rl_rvo_nav.policy_test.post_train import post_train
import argparse
import os
from os.path import dirname, abspath

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

parser = argparse.ArgumentParser(description='policy test')
parser.add_argument('--policy_type', default='drl')
parser.add_argument('--model_path', default='policy_train/model_save')
parser.add_argument('--model_name', default='r4_0/r4_0_250.pt')  #   policy_dict=False    
# parser.add_argument('--model_name', default='r4_0/r4_0_check_point_250.pt')  with check point --> policy_dict=True
parser.add_argument('--arg_name', default='r4_0/r4_0')
parser.add_argument('--world_name', default='policy_test_world.yaml')  # policy_test_world_lines.yaml
parser.add_argument('--render', action='store_true')
parser.add_argument('--robot_number', type=int, default='4')
parser.add_argument('--num_episodes', type=int, default='100')
parser.add_argument('--dis_mode', type=int, default='3')  # 3 circle, 2 random, 5 for corridor
parser.add_argument('--save', action='store_true')
parser.add_argument('--full', action='store_true')
parser.add_argument('--show_traj', action='store_true')
parser.add_argument('--policy_dict', action='store_true')
parser.add_argument('--once', action='store_true')

policy_args = parser.parse_args()

cur_path = Path(__file__).parent.parent 

# model_base_path = str(cur_path) + '/' + policy_args.model_path
model_base_path = dirname(dirname(abspath(__file__))) + '/' + policy_args.model_path
args_path = model_base_path + '/' + policy_args.arg_name

# args from train
r = open(args_path, 'rb')
args = pickle.load(r) 

if policy_args.policy_type == 'drl':
    fname_model = model_base_path + '/' + policy_args.model_name 
    policy_name = 'drl_rvo'
    
env = gym.make('mrnav-v1', world_name=policy_args.world_name, robot_number=policy_args.robot_number, neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, robot_init_mode=policy_args.dis_mode, env_train=False, random_bear=args.random_bear, random_radius=args.random_radius, reward_parameter=args.reward_parameter, goal_threshold=0.2, full=policy_args.full)

policy_name = policy_name + '_' + str(policy_args.robot_number) + '_dis' + str(policy_args.dis_mode)

pt = post_train(env, num_episodes=policy_args.num_episodes, reset_mode=policy_args.dis_mode, render=policy_args.render, std_factor=0.00001, acceler_vel=1.0, max_ep_len=300, neighbors_region=args.neighbors_region, neighbor_num=args.neighbors_num, args=args, save=policy_args.save, show_traj=policy_args.show_traj, figure_format='eps')
pt.policy_test(policy_args.policy_type, fname_model, policy_name, result_path=str(cur_path), result_name='/result.txt', figure_save_path=cur_path / 'figure' , ani_save_path=cur_path / 'gif', policy_dict=policy_args.policy_dict, once=policy_args.once)
