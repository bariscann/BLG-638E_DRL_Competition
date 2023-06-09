from random import randint,random
import time
from .utilities import *


# -------------------BOT CLASS------------------- #
# Bot class that moves units randomly on the map
# Creates new truck units with random prob < 0.05
class RandomAgent:
    def __init__(self,team,action_lenght):
        self.team = team
        self.enemy_team = (team+1)%2
        self.action_lenght = action_lenght

    def decode_state(self, state):
        score = state['score']
        turn = state['turn']
        max_turn = state['max_turn']
        units = state['units']
        hps = state['hps']
        bases = state['bases']
        res = state['resources']
        load = state['loads']

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
                        'tag': tagToString[units[self.team][i][j]],
                        'hp': hps[self.team][i][j],
                        'location': (i,j),
                        'load': load[self.team][i][j]
                    }
                    )
                if units[self.enemy_team][i][j]<6 and units[self.enemy_team][i][j] != 0:
                    self.enemy_units.append(
                    {   
                        'unit': units[self.enemy_team][i][j],
                        'tag': tagToString[units[self.enemy_team][i][j]],
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

    def action(self, state):
        self.decode_state(state)

        movement = []
        target = []
        location = []

        # -------------------UNIT MOVEMENT------------------- #
        # Units are randomly moving on the maps
        for unit in self.my_units:
            location.append(unit['location'])
            movement.append(randint(0,6))
            unit_target = randint(0,len(self.enemy_units)-1)
            target.append(self.enemy_units[unit_target]['location'])

        # -------------------NEW UNIT CREATION------------------- #
        train = 0 # Don't create any new unit on the map
        
        if random()<0.05:
            train = 1 # With random prob create new truck unit

        return (location, movement, target, train)
