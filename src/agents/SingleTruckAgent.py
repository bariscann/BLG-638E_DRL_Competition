from random import randint, random
import copy
from .utilities import *


# Truck moves to collect resource greedly
class TruckAgent:
    def __init__(self, team, action_lenght):
        self.team = team
        self.enemy_team = (team+1) % 2
        self.action_lenght = action_lenght

    def action(self, state):
        """
            Args:
            Returns:    
                An action tuple strcuture consisting of location, movement, target, train
                location: Locations of the units on the map
                movement: Desired movement for the units
                target: Targets of the units of the map, next state
                train: Train additional units on the map.
                Tuple[location: List[Tuple[(x,y)]], movement: List[int],
                      target: List[Tuple[(x,y)]], train: int]
        """
        self.y_max, self.x_max = state['resources'].shape
        
        decoded = decodeState(state)

        self.my_units = decoded[self.team]
        self.enemy_units = decoded[self.enemy_team]
        self.my_base = decoded[self.team + 2]
        self.enemy_base = decoded[self.enemy_team + 2]
        self.resources = decoded[4]

        location = []
        movement = []
        target = []
        counter = {"Truck":0} # Count for units
        
        for unit in self.my_units:
            counter[unit['tag']]+=1
            location.append(unit['location'])
            unt_pos = [unit['location'][0], unit['location'][1]]

            # TRUCK AGENT        
            if unit['tag'] == 'Truck':
                dis = 999
                target_pos = None
                unt_pos = [unit['location'][0], unit['location'][1]]
                
                if unit['load'] > 0: # Max truck capacity = 3
                    target_pos = [self.my_base[0], self.my_base[1]]
                elif len(self.resources) > 0: # Number of resources on the map  
                    for res in self.resources:
                        res_pos = [res[0], res[1]]
                        dist_tmp = getDistance(unt_pos, res_pos)
                        res_busy = False
                        
                        for u in self.my_units+self.enemy_units:
                            if u['location'][0] == res_pos[0] and u['location'][1] == res_pos[1] and not u['unit'] == unit['unit']:
                                res_busy = True
                                break
                        if dist_tmp < dis and not res_busy:
                            dis = dist_tmp
                            target_pos = res_pos
                else:
                    target_pos = unt_pos


                if target_pos is None:
                    target_pos = unt_pos

                # POSSIBLE 7 actions on the hexagonal map
                possible_actions = []

                for m_action in range(7):
                    move_x, move_y = getMovement(unt_pos, m_action)
                    act_pos = [unt_pos[0] + move_y, unt_pos[1] + move_x]
                    if act_pos[0] < 0 or act_pos[1] < 0 or act_pos[0] > self.y_max - 1 or act_pos[1] > self.x_max-1:
                        act_pos = [unt_pos[0] ,unt_pos[1]]
                    possible_actions.append([getDistance(target_pos, act_pos), target_pos, act_pos, m_action])
                possible_actions.sort()

                if random()<0.20:
                    movement.append(copy.copy(randint(1,6)))
                else:
                    movement.append(copy.copy(possible_actions[0][-1]))
                target.append(copy.copy(target_pos))

            else:
                movement.append(2)
                target.append([0,0])
        
        # NEW UNIT CREATION
        train = 0 # Don't create any new unit on the map

        # Create new units based on condition
        if state["score"][self.team]>state["score"][self.enemy_team]+2:
            if counter["Truck"]<2: # Create new truck unit
                train = stringToTag["Truck"]

        # Dimension check with assertions based on Template Agent
        assert len(movement) == self.action_lenght and len(target) == self.action_lenght

        return (location, movement, target, train)
