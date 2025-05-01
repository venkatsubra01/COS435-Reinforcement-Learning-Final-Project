from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mabrax import mabrax_env
from typing import Tuple
from jaxmarl.wrappers.baselines import LogWrapper

class MABraxAntSoccer(Env):
    def __init__(self):
        self.env = mabrax_env.AntSoccer()  # Use AntSoccer instead of Ant
        self.env = LogWrapper(self.env)

    def get_obs(self, state, obs, target):
        qpos = state.pipeline_state.q[:-4]  # Exclude target and object q
        qvel = state.pipeline_state.qd[:-4]  # Exclude target and object qd
        obs_arr = jp.stack([obs[a] for a in self.env.agents])
        print(self.env._env.env)
        print(vars(self.env._env.env))
        object_position = state.pipeline_state.x.pos[self.env._env.env._object_idx][:2]  # Add object position
        target_pos = state.pipeline_state.x.pos[-1][:2]  # Add target position
        return jp.concatenate(
            (obs_arr, jp.repeat(qpos[None, :], self.env.num_agents, axis=0),
             jp.repeat(qvel[None, :], self.env.num_agents, axis=0),
             jp.repeat(object_position[None, :], self.env.num_agents, axis=0),
             jp.repeat(target_pos[None, :], self.env.num_agents, axis=0)),
            axis=1
        )

    @property
    def observation_size(self) -> int:
        # print(self.env.observation_spaces)
        # print(self.env.observation_spaces[self.env.agents[0]].shape)
        # idk why but it's 51
        # return self.env.observation_spaces[self.env.agents[0]].shape[0] + 8  # Add 4 for object and 4 for target
        return 51

    @property
    def action_size(self) -> int:
        return self.env.action_spaces[self.env.agents[0]].shape[0]

    @property
    def backend(self) -> str:
        raise Exception("This environment does not use a brax backend")

    def reset(self, rng: jax.Array) -> State:
        """ Resets the environment to an initial state. """
        key1, key2, key3 = jax.random.split(rng, 3)
        
        # Reset the internal MPE environment
        obs, state = self.env.reset(key1)
        
        _, target = self._random_target(key3)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        info = {"seed": 0, "mpe_key": key2, "target": target, 'returned_episode_lengths': jp.zeros(self.env.num_agents),
                'returned_episode': jp.zeros(self.env.num_agents).astype(bool), 'returned_episode_returns': jp.zeros(self.env.num_agents),
                'episode_returns': jp.zeros(self.env.num_agents)}

        state = State(state, self.get_obs(state.env_state, obs, target), reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """ Run one timestep of the environment's dynamics. """
        key, key_s = jax.random.split(state.info["mpe_key"], 2)
        
        # generate action dict for adversaries and good agents
        actions = {agent: action[i] for i, agent in enumerate(self.env.agents)}

        pipeline_state = state.pipeline_state.replace(returned_episode_returns=state.info["returned_episode_returns"],
                                                      episode_returns=state.info["episode_returns"],
                                                      returned_episode_lengths=state.info["returned_episode_lengths"])
        
        # step internal environment
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, state.pipeline_state, actions)
        # process obs into our format
        obs = self.get_obs(mpe_state.env_state, obs, state.info["target"])
        
        # set trajectory id to differentiate between episodes
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]

        # update info
        info = {"seed": seed, "mpe_key": key}
        state.info.update(info)
        state.info.update({'returned_episode_lengths': infos['returned_episode_lengths'], 
                           'returned_episode': infos['returned_episode'], 
                           'returned_episode_returns': infos['returned_episode_returns']})
        state.info.update({"episode_returns": mpe_state.episode_returns})

        # update metrics
        dist = jp.linalg.norm(mpe_state.env_state.pipeline_state.q[:2] - state.info["target"])
        # success = jp.array(dist < self.env.goal_reach_thresh, dtype=float)  # Use goal_reach_thresh
        success = jp.array(dist < 0.5, dtype=float)  # Use goal_reach_thresh
        success_easy = jp.array(dist < 2.0, dtype=float)
        state.metrics.update({"dist": dist, "success": success, "success_easy": success_easy})

        # vel_to_target = (state.metrics["dist"] - dist) / self.dt
        # reward = 10 * vel_to_target + _ - ctrl_cost - contact_cost if self.env.dense_reward else success  # Dense reward logic
        reward, _ = jp.zeros(2)
        return state.replace(
            pipeline_state=mpe_state, obs=obs, reward=reward, done=list(dones.values())[0].astype(jp.float32)
        )
        
    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 10
        # ang = jp.pi * 2.0 * jax.random.uniform(rng2)
        ang = jp.pi * 0.5 
        
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        return rng, jp.array([target_x, target_y])
