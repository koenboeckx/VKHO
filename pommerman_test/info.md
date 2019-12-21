# Observations

| Feature                       | Dimensions            | 
| ------------------------------|:---------------       | 
| Board (walls, agents, etc.)   | 11x11 Integers        | 
| Bomb blast strength           | 11x11 Integers        |
| Bomb life                     | 11x11 Integers        |
| Ammo                          | 1 integer             |
| Blast strength                | 1 integer             |
| Position                      | 2 integers in [0, 10] |
| Can kick                      | 1 integer in [0,1]    |
| Teammate                      | 1 integer in [9, 13]  |
| Enemies                       | 3 integers in [9, 13] |

**Interpretation**

The 11x11 board is a numpy array where each value corresponds:

| Value | Meaning       | Value | Meaning   |
|-------|-----------    |-------|-----------
| 0     | Passage       | 7     | Increase range power-up |
| 1     | Rigid Wall    | 8     | Can-Kick power-up (can kick bombs by touching them) |
| 2     | Wooden Wall   | 9     | AgentDummy    |
| 3     | Bomb          | 10    | Agent0        |
| 4     | Flames        | 11    | Agent1        |
| 5     | Fog (partially observable setting) | 12    | Agent2        |
| 6     | Extra Bomb power-up (adds ammo)   | 13    | Agent3        |


________________
# Actions

| Action | Meaning |
|--------|---------|
| 0 | Stop |
| 1 | Up |
| 2 | Left |
| 3 | Down |
| 4 | Right |
| 5 | Bomb |

_____________________
# Comments
1. When a destructible wall is destroyed, there is a chance to reveal one of three different power-ups: Increased ammo (6), increased blast range (7) or the ability to kick bombs (8)
1. https://github.com/MultiAgentLearning/playground
1. https://github.com/rwightman/pytorch-pommerman-rl
1. https://github.com/eugene/pommerman
1. https://github.com/tambetm/pommerman-baselines
1. For data visualization: https://github.com/facebookresearch/visdom