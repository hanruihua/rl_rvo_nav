from matplotlib.pyplot import figure
import numpy as np
import argparse
import gym
import gym_env
import pickle
from torch import nn
from rl_rvo_nav.policy_train.multi_ppo import multi_ppo
from rl_rvo_nav.policy.policy_rnn_ac import rnn_ac
from pathlib import Path
import os
import cProfile
import shutil

cur_path = Path(__file__).parent
world_path = 'train_world.yaml'
world_abs_path = str(cur_path/'train_world.yaml')

r_num =4
mode = 3
counter = 0
con_train = ''

if mode == 2:
    robot_num = '/rr' + str(r_num)
else:
    robot_num = '/r' + str(r_num)

model_save_path = str(cur_path.parent/ 'policy_test'/'model_save') + robot_num +'_{}'
save_name = robot_num + '_{}'

while os.path.isfile( (model_save_path.format(counter)+save_name.format(counter)) ):
    counter+=1

model_save_path = model_save_path.format(counter)
figure_save_path = Path(model_save_path) / 'figure'
save_name = save_name.format(counter)
ani_save_path = Path(model_save_path) / 'ani'

load_name = '/r4_0'
load_fname = str(cur_path/'model_save') + load_name + load_name + '_check_point_500.pt'

parser = argparse.ArgumentParser(description='drl rvo parameters')

par_env = parser.add_argument_group('par env', 'environment parameters') 
par_env.add_argument('--env_name', default='mrnav-v1')
par_env.add_argument('--world_path', default=world_path)
par_env.add_argument('--robot_number', type=int, default=r_num)
par_env.add_argument('--init_mode', default=mode)
par_env.add_argument('--reset_mode', default=mode)
par_env.add_argument('--obs_mode', type=int, default=0)
par_env.add_argument('--reward_mode',type=int, default=1)
par_env.add_argument('--mpi', default=False)

par_env.add_argument('--neighbors_region', default=4)
par_env.add_argument('--neighbors_num', type=int, default=5)   
par_env.add_argument('--reward_parameter', type=float, default=(3.0, 0.3, 0.0, 6.0, 0.3, 3.0, -20, 20), nargs='+')
par_env.add_argument('--env_train', default=True)
par_env.add_argument('--random_bear', default=True)
par_env.add_argument('--random_radius', default=False)
par_env.add_argument('--full', default=False)

par_policy = parser.add_argument_group('par policy', 'policy parameters') 
par_policy.add_argument('--policy_mode', default='rnn') # transformer / rnn
par_policy.add_argument('--state_dim', default=6)
par_policy.add_argument('--rnn_input_dim', default=8)
par_policy.add_argument('--rnn_hidden_dim', default=256)
par_policy.add_argument('--trans_input_dim', default=8)
par_policy.add_argument('--trans_max_num', default=10)
par_policy.add_argument('--trans_nhead', default=1)
par_policy.add_argument('--trans_mode', default='attn')
par_policy.add_argument('--hidden_sizes_ac', default=(256, 256))
par_policy.add_argument('--drop_p', type=float, default=0)
par_policy.add_argument('--hidden_sizes_v', type=tuple, default=(256, 256))  # 16 16
par_policy.add_argument('--activation', default=nn.ReLU)
par_policy.add_argument('--output_activation', default=nn.Tanh)
par_policy.add_argument('--output_activation_v', default=nn.Identity)
par_policy.add_argument('--use_gpu', type=bool, default=True)
par_policy.add_argument('--rnn_mode', default='biGRU')   # LSTM

par_train = parser.add_argument_group('par train', 'train parameters') 
par_train.add_argument('--pi_lr', type=float, default=4e-6)
par_train.add_argument('--vf_lr', type=float, default=5e-5)
par_train.add_argument('--train_epoch', type=int, default=10000)
par_train.add_argument('--steps_per_epoch', type=int, default=500)
par_train.add_argument('--max_ep_len', default=150)
par_train.add_argument('--gamma', default=0.99)
par_train.add_argument('--lam', default=0.97)
par_train.add_argument('--clip_ratio', default=0.2)
par_train.add_argument('--train_pi_iters', default=50)
par_train.add_argument('--train_v_iters', default=50)
par_train.add_argument('--target_kl',type=float, default=0.05)
par_train.add_argument('--render', default=True)
par_train.add_argument('--render_freq', default=50)
par_train.add_argument('--con_train', default=False)
par_train.add_argument('--seed', default=7)
par_train.add_argument('--save_freq', default=50)
par_train.add_argument('--save_figure', default=False)
par_train.add_argument('--figure_save_path', default=figure_save_path)
par_train.add_argument('--save_path', default=model_save_path)
par_train.add_argument('--save_name', default=save_name)
par_train.add_argument('--load_fname', default=load_fname)
par_train.add_argument('--save_result', type=bool, default=True)
par_train.add_argument('--lr_decay_epoch', type=int, default=1000)
par_train.add_argument('--max_update_num', type=int, default=10)

args = parser.parse_args(['--train_epoch', '10000', '--con_train', con_train, '--pi_lr', '4e-6', '--vf_lr', '5e-5'])

env = gym.make(args.env_name, world_name=args.world_path, robot_number=args.robot_number, neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, robot_init_mode=args.init_mode, env_train=args.env_train, random_bear=args.random_bear, random_radius=args.random_radius, reward_parameter=args.reward_parameter, obs_mode=args.obs_mode, reward_mode=args.reward_mode, full=args.full)

test_env = gym.make(args.env_name, world_name=args.world_path, robot_number=args.robot_number, neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, robot_init_mode=args.init_mode, env_train=False, random_bear=args.random_bear, random_radius=args.random_radius, reward_parameter=args.reward_parameter, plot=False, obs_mode=args.obs_mode, reward_mode=args.reward_mode)

if args.policy_mode == 'rnn':
    policy = rnn_ac(env.observation_space, env.action_space, args.state_dim, args.rnn_input_dim, args.rnn_hidden_dim, 
                    args.hidden_sizes_ac, args.hidden_sizes_v, args.activation, args.output_activation, 
                    args.output_activation_v, args.use_gpu, args.rnn_mode, args.drop_p)

ppo = multi_ppo(env, policy, args.pi_lr, args.vf_lr, args.train_epoch, args.steps_per_epoch, args.max_ep_len, args.gamma, args.lam, args.clip_ratio, args.train_pi_iters, args.train_v_iters, args.target_kl, args.render, args.render_freq, args.con_train,  args.seed, args.save_freq, args.save_figure, args.save_path, args.save_name, args.load_fname, args.use_gpu, args.reset_mode, args.save_result, counter, test_env, args.lr_decay_epoch, args.max_update_num, args.mpi, args.figure_save_path)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

f = open(args.save_path+args.save_name, 'wb')
pickle.dump(args, f)
f.close()

with open(args.save_path+args.save_name+'.txt', 'w') as p:
    print(vars(args), file=p)
p.close()
# print(args)

shutil.copyfile(world_abs_path, args.save_path+args.save_name+'_world.yaml')

# cProfile.run('ppo.training_loop()', sort=2)
ppo.training_loop()
# env.env_gym.world_plot.create_animate(figure_save_path, ani_save_path, ani_name='drl_train_'+str(r_num), keep_len=30, rm_fig_path=True)
