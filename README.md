# AgentsOfGlory
An strategy simulation game for competing agents.

## How to Install
`git clone https://github.com/captanc/AgentsOfGlory.git`

Make sure that python packages in `requirements.txt` are installed. You can create a conda environment from that file directly.

## How to Run

### Single to Run

`python src/agentsofglory.py "map" "agent_blue" "agent_red"`

Example run: `python src/agentsofglory.py ResourceRiver SimpleAgent RandomAgent`

### Tournament Run

Two types of tournament are available:
- Elimination
- League

`python src/tournament.py "tournament"`

Example run: `python src/tournament.py tournament_elimination`

## Gym

### State

State representation is the knowladge from the game that is being send to agents when it is their turn.

State includes following members:
`score = int, shape(2)`
`turn = int, shape(1)`
`max_turn = int, shape(1)`
`units = int, shape(y,x)`
`hps = int, shape(y,x)`
`bases = int, shape(y,x)`
`res = int, shape(y,x)`
`load = int, shape(y,x)`

Example state decoding: 
`score = state['score']`
`turn = state['turn']`
`max_turn = state['max_turn']`
`units = state['units']`
`hps = state['hps']`
`bases = state['bases']`
`res = state['resources']`
`load = state['loads']`

### Action

`[location, movement, target]`

Location (y,x) Tuple

Movement: 0-6 where 1-6 are movement action with direction given in the image, 0 is the shoot or collect actions![Movement](documentation/images/action.jpg)

Target: (y,x) Tuple

Order of processing the action is done by the order returned from the agent.

## Game

### Grid World

Grid world representation: ![Grid_world](documentation/images/state.png)

### Units

The game involves four units:
- HeavyTank
- LightTank
- Truck
- Drone

Unit rules can be changed in `data/config/rules.yaml`

## Maps

Maps are defined as yaml files in `data/config`.

There are currently two maps available:
- demo
- ResourceRiver

If you wish to create your own map, please take `demo.yaml` as a layout.

## Agents

Agents are located at `src/agents` directory. Each Agent has a file with its name and a class definition with its name. 

Each Agent should have following functions:
- `__init__((self,team,action_lenght)`
- `action(self, state)`

There are currently four agents available:
-HumanAgent
-RandomAgent
-SimpleAgent
-BlockingAgent(Not Implemented)

If you wish to create your own agent, please use `TemplateAgent` as a layout.

### HumanAgent

HumanAgent is a special agent with testing purposes. This agent gets each action for each unit from user from console.

### RandomAgent

RandomAgent generates actions randomly.

### SimpleAgent

SimpleAgent only uses `Truck` and `LightTank` units. Other units constantly tries to move upward. `Truck` units tries to reach closest resource and when collected a resource moves back to base. `LightTank` units moves to its target if distance is greater than 2 else shoots. Trucks are selected as target if available.

### BlockingAgent

Not İmplemented

BlockingAgent blocks enemy base to prevent other team to score.