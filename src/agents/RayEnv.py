from agents.BaseLearningGym import BaseLearningAgentGym
import gym
from gym import spaces
import numpy as np
import yaml
from game import Game
from utilities import forced_anchor, necessary_obs, decode_location, reward_shape



def read_hypers():
    with open(f"/home/captanc/AgentsOfGlory/data/config/TrainSingleTruckSmall.yaml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict


class RayEnv(BaseLearningAgentGym):

    def __init__(self, args, agents):
        super().__init__() 
        configs = read_hypers()
        self.game = Game(args, agents)
        self.team = 0
        self.enemy_team = 1
        self.tagToString = {
            1: "Truck",
            2: "LightTank",
            3: "HeavyTank",
            4: "Drone",
        }

        self.height = configs['map']['y']
        self.width = configs['map']['x']
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.nec_obs = None
        self.observation_space = spaces.Box(
            low=0,
            high=150,
            shape=(208,),
            dtype=np.uint8
        )
        self.action_space = spaces.MultiDiscrete([self.width, self.height, 7, self.width, self.height, 5])




    def setup(self, obs_spec, action_spec):
        self.observation_space = obs_spec
        self.action_space = action_spec

    def reset(self):
        self.episodes += 1
        self.steps = 0
        state = self.game.reset()
        self.nec_obs = state
        return self.decode_state(state)

    def decode_state(self, obs):
        turn = obs['turn']
        max_turn = obs['max_turn']
        units = obs['units']
        hps = obs['hps']
        bases = obs['bases']
        score = obs['score']
        res = obs['resources']
        
        load = obs['loads']
        self.y_max, self.x_max = res.shape
        self.my_units = []
        self.enemy_units = []
        self.resources = []
        for i in range(self.y_max):
            for j in range(self.x_max):
                if units[self.team][i][j]<6 and units[self.team][i][j] != 0:
                    self.my_units.append(
                    {   
                        'unit': units[self.team][i][j],
                        'tag': self.tagToString[units[self.team][i][j]],
                        'hp': hps[self.team][i][j],
                        'location': (i,j),
                        'load': load[self.team][i][j]
                    }
                    )
                if units[self.enemy_team][i][j]<6 and units[self.enemy_team][i][j] != 0:
                    self.enemy_units.append(
                    {   
                        'unit': units[self.enemy_team][i][j],
                        'tag': self.tagToString[units[self.enemy_team][i][j]],
                        'hp': hps[self.enemy_team][i][j],
                        'location': (i,j),
                        'load': load[self.enemy_team][i][j]
                    }
                    )
                if res[i][j]==1:
                    self.resources.append((i,j))
                if bases[self.team][i][j]:
                    self.my_base = (i,j)
                if bases[self.enemy_team][i][j]:
                    self.enemy_base = (i,j)
        
        unitss = [*units[0].reshape(-1).tolist(), *units[1].reshape(-1).tolist()]
        hpss = [*hps[0].reshape(-1).tolist(), *hps[1].reshape(-1).tolist()]
        basess = [*bases[0].reshape(-1).tolist(), *bases[1].reshape(-1).tolist()]
        ress = [*res[0].reshape(-1).tolist(), *res[1].reshape(-1).tolist()]
        loads = [*load[0].reshape(-1).tolist(), *load[1].reshape(-1).tolist()]

        
        state = (*score.tolist(), turn, max_turn, *unitss, *hpss, *basess, *ress, *loads)
        return np.array(state, dtype=np.int)


    def take_action(self, action):
        """
        action_space'in ilk 2 elemanı location koordinatları, 
        3.sü movement 
        4 ve 5 target koordinatları
        6. ise train
        """
        location = [[action[0], action[1]]]
        locations = decode_location(self.my_units)
        if location not in locations:
            location = [locations[0]]
        movement = [action[2]]
        movement = forced_anchor(movement, self.nec_obs)
        target = [[action[3], action[4]]]
        train = 0

        action = [location, movement, target, train]
        print(action)
        return action

    def step(self, action):
        action = self.take_action(action)
        next_state, _, done =  self.game.step(action)
        reward = reward_shape(self.nec_obs)
        info = {}
        self.steps += 1
        self.reward += reward
        self.nec_obs = next_state
        return self.decode_state(next_state), reward, done, info

    def render(self,):
        return None

    def close(self,):
        return None