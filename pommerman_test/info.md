**Observations**

| Feature                       | Dimensions            | 
| ------------------------------|:---------------       | 
| Board (walls, agents, etc.)   | 11x11 Integers        | 
| Bomb blast strength           | 11x11 Integers        |
| Bomb life                     | 11x11 Integers        |
| Ammo                          | 1 integer             |
| Blast strength                | 1 integer             |
| Position                      | 2 integers in [0, 10] |
| Can kick                      | 1 integer in [0,1]    |
| Teammate                      | 1 integer in [-1, 3]  |
| Enemies                       | 3 integers in [-1, 3] |

________________
**Actions**

| Action | Meaning |
|--------|---------|
| 0 | Stop |
| 1 | Up |
| 2 | Left |
| 3 | Down |
| 4 | Right |
| 5 | Bomb |

_____________________
**Comments***
1. When a destructible wall is destroyed, there is a chance to reveal one of three different power-ups: Increased ammo, increased blast range or the ability
1. https://github.com/MultiAgentLearning/playground
1. https://github.com/rwightman/pytorch-pommerman-rl
1. https://github.com/eugene/pommerman
1. https://github.com/tambetm/pommerman-baselines
1. For data visualization: https://github.com/facebookresearch/visdom