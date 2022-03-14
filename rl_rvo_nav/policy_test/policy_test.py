from drl_rvo_nav_full.policy_test.post_train import post_train
import gym
import gym_env
from pathlib import Path
import pickle

epoch = 250

cur_path = Path(__file__).parent
save_name = 'r10_0'
save_path_string = str(cur_path /'model_save'/save_name/save_name)

fname_model = save_path_string +'_check_point_'+ str(epoch)+ '.pt' 
arg_model = save_path_string

world_path = Path('model_save')
world_path_str = str(world_path/save_name/save_name) + '_world.yaml'

r = open(arg_model, 'rb')
args = pickle.load(r) 

env = gym.make('mrnav-v0', world_name=world_path_str, robot_number=4, neighbors_region=args.neighbors_region, neighbors_num=3, robot_init_mode=3, env_train=False, random_bear=True, random_radius=args.random_radius, reward_parameter=args.reward_parameter, obs_mode=args.obs_mode, reward_mode=args.reward_mode)

pt = post_train(env, num_episodes=100, reset_mode=2, render=True, std_factor=0.01, acceler_vel=0.5, max_ep_len=300)
pt.policy_test('drl', fname_model, save_name, result_path=str(cur_path), result_name='/result.txt')

