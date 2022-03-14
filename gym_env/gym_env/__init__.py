from gym.envs.registration import register

register(
    id='mrnav-v1',
    entry_point='gym_env.envs:mrnav',
)
