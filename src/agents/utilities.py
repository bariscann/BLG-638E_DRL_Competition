from PIL.Image import new
from pandas.core import base
from yaml import load
import numpy as np
import copy
import time
from statistics import mode

tagToString = {
    1: "Truck",
    2: "LightTank",
    3: "HeavyTank",
    4: "Drone",
    }
stringToTag = {
    "Truck": 1,
    "LightTank": 2,
    "HeavyTank": 3,
    "Drone": 4,
    }
shoot_range = {
    "LightTank": 2,
    "HeavyTank": 2,
    "Drone": 1,
}

movement_grid = [[(0, 0), (-1, 0), (0, -1), (1, 0), (1, 1), (0, 1), (-1, 1)],
[(0, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 0)]]

# double_distance = [(movement_grid[0][i][0] + movement_grid[1][i][0], movement_grid[0][i][1] + movement_grid[1][i][1]) for i in range(7)]
# other_double_distance = [(0,0), (0,-2), (-1,-1), (-1,1), (0,2), (1,1), ()]

def getMovement(unit_position, action):
    return movement_grid[unit_position[1] % 2][action]

"""
def getActionToEscape(ally_loc, enemy_loc):
    distance = getDistance(ally_loc, enemy_loc)
    dis_tuple = (ally_loc[0] - enemy_loc[0], ally_loc[1] - enemy_loc[1])
    if distance == 2:
        if dis_tuple in double_distance:
            return double_distance.index(dis_tuple)
        else:
            return reversed_double_distance.index(dis_tuple)
    else:
        right_movement_list = movement_grid[enemy_loc[1] % 2]
        # return right_movement_list.index((ally_loc[0] - enemy_loc[0], ally_loc[1] - enemy_loc[1]))
        return right_movement_list.index((ally_loc[1] - enemy_loc[1], ally_loc[0] - enemy_loc[0]))
"""

def decodeState(state):
    score = state['score']
    turn = state['turn']
    max_turn = state['max_turn']
    units = state['units']
    hps = state['hps']
    bases = state['bases']
    res = state['resources']
    load = state['loads']
    
    blue = 0
    red = 1
    y_max, x_max = res.shape
    blue_units = []
    red_units = []
    resources = []
    blue_base = None
    red_base = None
    for i in range(y_max):
        for j in range(x_max):
            if units[blue][i][j] < 6 and units[blue][i][j] != 0 and hps[blue][i][j]>0:
                blue_units.append(
                    {
                        'unit': units[blue][i][j],
                        'tag': tagToString[units[blue][i][j]],
                        'hp': hps[blue][i][j],
                        'location': (i, j),
                        'load': load[blue][i][j]
                    }
                )
            if units[red][i][j] < 6 and units[red][i][j] != 0 and hps[red][i][j]>0:
                red_units.append(
                    {
                        'unit': units[red][i][j],
                        'tag': tagToString[units[red][i][j]],
                        'hp': hps[red][i][j],
                        'location': (i, j),
                        'load': load[red][i][j]
                    }
                )
            if res[i][j] == 1:
                resources.append((i, j))
            if bases[blue][i][j]:
                blue_base = (i, j)
            if bases[red][i][j]:
                red_base = (i, j)
    return [blue_units, red_units, blue_base, red_base, resources]


def getDistance(pos_1, pos_2):
    if pos_1 == None or pos_2 == None:
        return 999
    pos1 = copy.copy(pos_1)
    pos2 = copy.copy(pos_2)
    shift1 = (pos1[1]+1)//2
    shift2 = (pos2[1]+1)//2
    pos1[0] -= shift1
    pos2[0] -= shift2
    distance = (abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1]) + abs(pos1[0]+pos1[1]-pos2[0]-pos2[1]))//2
    return distance


def decode_location(my_units):
    locations = []
    for unit in my_units:
        locations.append(unit["location"])
    return locations



def enemy_locs(obs, team):
    enemy_units = obs['units'][(team+1) % 2]
    enemy_list1 = np.argwhere(enemy_units != -1)
    enemy_list1 = set((tuple(i) for i in enemy_list1))
    enemy_list2 = np.argwhere(enemy_units != 0)
    enemy_list2 = set((tuple(i) for i in enemy_list2))
    return np.asarray(list(enemy_list1.intersection(enemy_list2)))


def ally_locs(obs, team):

    ally_units = obs['units'][team]
    ally_list1 = np.argwhere(ally_units != -1)
    ally_list1 = set((tuple(i) for i in ally_list1))
    ally_list2 = np.argwhere(ally_units != 0)
    ally_list2 = set((tuple(i) for i in ally_list2))

    return list(ally_list1.intersection(ally_list2))

def truck_locs(obs, team):
    hps = np.array(obs['hps'][team])
    ally_units = np.array(obs['units'][team])
    ally_units[hps<1] = 0
    ally_list = np.argwhere(ally_units == 1)
    ally_list = ally_list.squeeze()

    return ally_list

def nearest_enemy(allied_unit_loc, enemy_locs):
    distances = []
    for enemy in enemy_locs:
        distances.append(getDistance(allied_unit_loc, enemy))
    nearest_enemy_loc = np.argmin(distances)

    return enemy_locs[nearest_enemy_loc]

def get_n_resource(obs):
    resources = obs['resources']
    resource_loc = np.argwhere(resources == 1)
    return len(resource_loc)

def multi_forced_anchor(movement, obs, team): # birden fazla truck için
    bases = obs['bases'][team]
    units = obs['units'][team]
    loads = obs['loads'][team]
    resources = obs['resources']
    hps = obs["hps"][team]
    score = obs['score']
    unit_loc = np.argwhere(units == 1)
    unit_loc = unit_loc.squeeze()
    base_loc = np.argwhere(bases == 1)
    base_loc = base_loc.squeeze()
    loaded_loc = np.argwhere(loads != 0)
    loaded_trucks = loads[loads != 0]
    resource_loc = np.argwhere(resources == 1)
    allies = ally_locs(obs, team)
    trucks = truck_locs(obs, team)

    for i,ally in enumerate(allies):
        if len(trucks) == 0 or i>6:
            break
        if isinstance(trucks[0], np.int64):
            trucks = np.expand_dims(trucks, axis=0)
        for truck in trucks:
            if (ally == truck).all():
                for reso in resource_loc:
                    if loads[truck[0], truck[1]].max() != 3 and (reso == truck).all():
                        movement[i] = 0
                    elif loads[truck[0], truck[1]].max() != 0 and (truck == base_loc).all():
                        movement[i] = 0
                    else:
                        continue
    return movement

def collect_resource(truck_loc, obs, team, movement):
    resources = obs['resources']
    resource_loc = np.argwhere(resources == 1)
    loads = obs['loads'][team]
    bases = obs['bases'][team]
    base_loc = np.argwhere(bases == 1)
    base_loc = base_loc.squeeze()
    for reso in resource_loc:
        if loads[truck_loc[0], truck_loc[1]].max() != 3 and (reso == truck_loc).all():
            return 0
        elif loads[truck_loc[0], truck_loc[1]].max() != 0 and (truck_loc == base_loc).all():
            return 0
    return movement

def forced_anchor(movement, obs, team_no):
    bases = obs['bases'][team_no]
    units = obs['units'][team_no]
    loads = obs['loads'][team_no]
    resources = obs['resources']
    unit_loc = np.argwhere(units == 1)
    unit_loc = unit_loc.squeeze()
    base_loc = np.argwhere(bases == 1)
    base_loc = base_loc.squeeze()
    resource_loc = np.argwhere(resources == 1)
    for reso in resource_loc:
        if (reso == unit_loc).all() and loads.max() != 3:
            movement = [0]
        else:
            continue
        if (reso == base_loc).all() and loads.max() != 0:
            movement = [0]
    return movement

def Shoot(obs, loc, team):
    enemy_units = obs['units'][(team+1) % 2]
    enemy_list = np.argwhere(enemy_units != 0)
    enemy_list = enemy_list.squeeze()


def point_blank_shoot(allied_unit_loc, enemy_locs):
    # yakında düşman varsa onun loc unu döndürüyor
    distances = []
    for enemy in enemy_locs:
        distances.append(getDistance(allied_unit_loc, enemy))
    if min(distances) <= 2:
        nearest_enemy_loc = np.argmin(distances)
        return enemy_locs[nearest_enemy_loc]
    else:
        return None

def getEnemiesCanShoot(allied_unit_loc, enemy_locs, type_of_unit, enemy_type_of_units):
    distances = []
    enemy_list = []
    for i, enemy in enumerate(enemy_locs):
        type_of_enemy = enemy_type_of_units[i]
        if (type_of_unit == stringToTag["HeavyTank"] and type_of_enemy == stringToTag["Drone"]) or type_of_enemy > 4:
            continue
        dist = getDistance(allied_unit_loc, enemy)
        if dist <= shoot_range[tagToString[type_of_unit]]:
            distances.append(dist)
            enemy_list.append(enemy)
    distances = np.array(distances)
    enemy_list = np.array(enemy_list)
    sorted_index = np.argsort(distances)
    enemy_list = enemy_list[sorted_index]
    return enemy_list

def necessary_obs(obs, team):
    ally_base = obs['bases'][team]
    enemy_base = obs['bases'][(team+1) % 2]
    ally_units = obs['units'][team]
    enemy_units = obs['units'][(team+1) % 2]
    ally_loads = obs['loads'][team]
    resources = obs['resources']

    ally_unit_loc = np.argwhere(ally_units == 1).squeeze()
    enemy_unit_loc = np.argwhere(enemy_units == 1).squeeze()
    ally_base_loc = np.argwhere(ally_base == 1).squeeze()
    enemy_base_loc = np.argwhere(enemy_base == 1).squeeze()
    resource_loc = np.argwhere(resources == 1)
    truck_load = [ally_loads.max(), 0]
    resource = [coo for coords in resource_loc for coo in coords]

    new_obs = [*ally_unit_loc.tolist(), *enemy_unit_loc.tolist(), *ally_base_loc.tolist(), *enemy_base_loc.tolist(), *resource, *truck_load]
    
    if len(new_obs) == 20:
        # print(new_obs)
        time.sleep(1)
    return new_obs

def reward_shape(obs, team):
    load_reward = 0
    unload_reward = 0
    bases = obs['bases'][team]
    units = obs['units'][team]
    loads = obs['loads'][team]
    resources = obs['resources']
    unit_loc = np.argwhere(units == 1)
    unit_loc = unit_loc.squeeze()
    base_loc = np.argwhere(bases == 1)
    base_loc = base_loc.squeeze()
    resource_loc = np.argwhere(resources == 1)
    for reso in resource_loc:
        if (reso == unit_loc).all() and loads.max() != 3:
            load_reward += 1
        else:
            continue
        if (reso == base_loc).all() and loads.max() != 0:
            unload_reward += 10

    return load_reward + unload_reward

def multi_reward_shape(obs, team): # Birden fazla truck için
    load_reward = 0
    unload_reward = 0
    enemy_load_reward = 0
    enemy_unload_reward = 0
    bases = obs['bases'][team]
    units = obs['units'][team]
    enemy_bases = obs['bases'][(team+1) % 2]
    enemy_units = obs['units'][(team+1) % 2]
    enemy_loads = obs['loads'][(team+1) % 2]
    loads = obs['loads'][team]
    resources = obs['resources']
    unit_loc = np.argwhere(units == 1)
    unit_loc = unit_loc.squeeze()
    base_loc = np.argwhere(bases == 1)
    base_loc = base_loc.squeeze()
    enemy_unit_loc = np.argwhere(enemy_units == 1)
    enemy_unit_loc = enemy_unit_loc.squeeze()
    enemy_base_loc = np.argwhere(enemy_bases == 1)
    enemy_base_loc = enemy_base_loc.squeeze()
    resource_loc = np.argwhere(resources == 1)
    enemy = enemy_locs(obs, team)
    ally = ally_locs(obs, team)
    trucks = truck_locs(obs, team)

    for truck in trucks:
        for reso in resource_loc:
            # print(reso,"RESOURCE")
            if not isinstance(truck, np.int64):
                # print(loads.shape, "load shape")
                # print(loads[truck[0], truck[1]].shape, "load at truck")
                # print(truck.shape, "Last Truck")
                if (reso == truck).all():
                    if loads[truck[0], truck[1]].max() != 3: 
                        load_reward += 10
            else:
                pass
            if not isinstance(truck, np.int64):
                if loads[truck[0], truck[1]].max() != 0 and (truck == base_loc).all():
                    unload_reward += 20

    harvest_reward = load_reward + unload_reward + enemy_load_reward + enemy_unload_reward
    return harvest_reward, len(enemy), len(ally)

def getTypeOfUnits(unit_locs, raw_state, team):
    units = np.array(raw_state['units'][team])
    types = []
    for x, y in unit_locs:
        types.append(units[x][y])
    return types

# ----------------------------------RULE FUNCTIONS---------------------------------- #
def train_rule(train, raw_state, team, th=0):
    loc_of_truck = truck_locs(raw_state, team)
    n_resource = get_n_resource(raw_state)
    enemy_units = raw_state['units'][(team+1)%2]
    enemy_n_htank = len(np.argwhere(enemy_units == 3))
    our_units = raw_state['units'][team]
    n_drones = len(np.argwhere(our_units == 4))
    # if n_resource > 0:
    #     n_truck = len(loc_of_truck)
    #     if n_truck <= th:
    #         train = stringToTag['Truck']
    if n_resource == 0 and train == stringToTag['Truck']:
        train = stringToTag['LightTank']
    if train != stringToTag['Truck'] and enemy_n_htank > n_drones:
        train = stringToTag['Drone']
        # train = stringToTag['Drone']
    return train

def movement_rule(movement, raw_state, team, locations, enemies, enemy_order):
    """ Movement rules for agents
    """
    # movement = multi_forced_anchor(movement, raw_state, team)

    x_max, y_max = raw_state["resources"].shape
    types_of_units = getTypeOfUnits(locations, raw_state, team)
    enemy_type_of_units = getTypeOfUnits(enemies, raw_state, (team+1)%2)
    
    types_of_units = np.array(types_of_units)
    locations = np.array(locations)
    sorted_index = np.argsort(types_of_units)
    locations = locations[sorted_index]
    types_of_units = types_of_units[sorted_index]

    unit_action_list = {1: [], 2: [], 3: [], 4:[]}
    already_deadth = []
    # TODO burada location 17 veya 34 ten fazla olabilir bu durumda patlayabilir.
    for i, (x,y) in enumerate(locations):
        type_of_unit = types_of_units[i]

        if i >= len(movement):
            movement.append(mode(unit_action_list[type_of_unit]))
            enemy_order.append([0,0])
        
        if type_of_unit > 4:
            movement[i] = 0
            continue
        # print(type_of_unit, type_of_unit)
        unit_action_list[type_of_unit].append(movement[i])
        movement_unit = movement[i]
        if type_of_unit == stringToTag['Truck']:
            movement[i] = collect_resource([x,y], raw_state, team, movement[i])
            enemy_loc = point_blank_shoot([x,y], enemies.tolist())
            if enemy_loc:
                movement[i] = movement_unit
        else:
            move_x, move_y = getMovement((x,y), movement_unit)
            act_pos = [x + move_y, y + move_x]
            if act_pos[0] < 0 or act_pos[1] < 0 or act_pos[1] > y_max-1 or act_pos[0] > x_max-1:
                act_pos = [x, y]
            
            # TODO can change as raw_state['terrain'][act_pos[0]][act_pos[1]] != 0 if there is no terrian greater than 3
            if type_of_unit == stringToTag['HeavyTank'] and raw_state['terrain'][act_pos[0]][act_pos[1]] != 0:
                    movement[i] = 0 # (movement_unit + 3)%6
            elif type_of_unit == stringToTag['LightTank'] and (raw_state['terrain'][act_pos[0]][act_pos[1]] == 2 or raw_state['terrain'][act_pos[0]][act_pos[1]] == 3):
                    movement[i] = 0 # (movement_unit + 3)%6
            # elif type_uf_unit == stringToTag['Drone'] and raw_state['terrain'][act_pos[0]][act_pos[1]] == 2:
            #         movement[i] = (movement_unit + 3)%6
            enemy_locs = getEnemiesCanShoot([x,y], enemies.tolist(), type_of_unit, enemy_type_of_units)
            for enemy_loc in enemy_locs:
                enemy_loc = enemy_loc.tolist()
                # if enemy_loc not in already_deadth:
                if True:
                    enemy_order[i] = enemy_loc
                    movement[i] = 0
                    already_deadth.append(enemy_loc)
                    break
        
    return locations, movement, enemy_order