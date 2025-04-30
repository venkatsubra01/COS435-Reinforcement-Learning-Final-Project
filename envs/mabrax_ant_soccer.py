from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mabrax import mabrax_env
from typing import Tuple
from jaxmarl.wrappers.baselines import LogWrapper

class MABraxAntSoccer(Env):
    def __init__(self):
        self.env = mabrax_env.AntSoccer()
        self.env = LogWrapper(self.env)

    def get_obs(self, state, obs, target):
        qpos = state.pipeline_state.q[:4]
        obs_arr = jp.stack([obs[a] for a in self.env.agents])
        return jp.concatenate((obs_arr, jp.repeat(qpos[None,:], self.env.num_agents, axis=0), 
                          jp.repeat(target[None,:], self.env.num_agents, axis=0)), axis=1)