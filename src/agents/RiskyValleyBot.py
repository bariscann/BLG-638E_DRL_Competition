from os import kill
from agents.BaseLearningGym import BaseLearningAgentGym
import gym
from gym import spaces
import pickle
import numpy as np
import yaml
from game import Game
from utilities import multi_forced_anchor, necessary_obs, decode_location, multi_reward_shape, enemy_locs, ally_locs, getDistance
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import run_experiments, register_env
from agents.RiskyValley import RiskyValley
from argparse import Namespace




class RiskyValleyBot(BaseLearningAgentGym):

    def __init__(self, team, action_lenght):
        self.team = team
        self.enemy_team = (team+1) % 2
        self.action_lenght = action_lenght
        agents = [None, "SimpleAgent"]
        args = Namespace(map="RiskyValley", render=False, gif=False, img=False)
    
        # ray.init()
        config= {"use_critic": True,
             "num_workers": 1,
             "use_gae": True,
             "lambda": 1.0,
             "kl_coeff": 0.2,
             "rollout_fragment_length": 200,
             "train_batch_size": 4000,
             "sgd_minibatch_size": 128,
             "shuffle_sequences": True,
             "num_sgd_iter": 30,
             "lr": 5e-5,
             "lr_schedule": None,
             "vf_loss_coeff": 1.0,
             "framework": "torch",
             "entropy_coeff": 0.0,
             "entropy_coeff_schedule": None,
             "clip_param": 0.3,
             "vf_clip_param": 10.0,
             "grad_clip": None,
             "kl_target": 0.01,
             "batch_mode": "truncate_episodes",
             "observation_filter": "NoFilter"}
        register_env("ray", lambda config: RiskyValley(args, agents))
        ppo_agent = PatchedPPOTrainer(config=config, env="ray")
        ppo_agent.restore(checkpoint_path="models/checkpoint_000200/checkpoint-200") # Modelin Bulunduğu yeri girmeyi unutmayın!
        self.policy = ppo_agent.get_policy()

    def action(self, raw_state):
        
        state = RiskyValley.just_decode_state(raw_state, self.team, self.enemy_team)
        actions, _, _ = self.policy.compute_single_action(state.astype(np.float32))
        location, movement, target, train = RiskyValley.just_take_action(actions, raw_state, self.team)
        return (location, movement, target, train)
        
class PatchedPPOTrainer(PPOTrainer):
    #@override(Trainable)
    def load_checkpoint(self, checkpoint_path: str) -> None:
        extra_data = pickle.load(open(checkpoint_path, "rb"))
        worker = pickle.loads(extra_data["worker"])
        worker = PatchedPPOTrainer.__fix_recursively(worker)
        extra_data["worker"] = pickle.dumps(worker)
        self._setstate_(extra_data)

    def __fix_recursively(data):
        if isinstance(data, dict):
            return {key: PatchedPPOTrainer.__fix_recursively(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [PatchedPPOTrainer.__fix_recursively(value) for value in data]
        elif data is None:
            return 0
        else:
            return data