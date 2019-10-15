1. Implement simple game:
    * symmetric (2 players with each 2 tanks)
    * only 8 possible actions:
        1. do_nothing
        1. aim1, i.e. aim at first agent of other team
        1. aim2, i.e. aim at second agent of other team
        1. fire -> hit only if target is close enough
        1. move_up
        1. move_down
        1. move_left
        1. move_right
    * (maybe) provide unlimited ammo
    * no fuel attribute
    * actions are executed simutaneously (a0, a1, a2, a3)
    * implement visulisation tool based on PyGame
    
1. Implement MCTS to develop non-trivial strategies for both players
    * centralised control with shared observations
    * consider global game state?
    * how to analyze (NE?, ESS?)
    
1. Deploy MARL algorithms to develop strategies
    * compare with MCTS results
    
1. Extend game to more realistic situations with
    * richer action space
    * more evolved sta te / observations (e.g. partial observability?)

1. Other ideas:
    * MCTS: reduce the game to turn-based game:
        * first execute team1's actions: (a0, a1, 0, 0)
        * evaluate game to see if team 1 won
        * then execute team2's actions:  (0, 0, a2, a3)
        * evaluate game to see if team 2 won
    -> otherwise, joint action space become too large (8**4 = 4096)
    * MCTS: based on FULL state information (thus not only individual observations)
    * MCTS flow graph created with https://www.lucidchart.com

1. Development summary:
    1. [15Oct19] An infinte loop in while not player.is_leaf()
        * rewrote envs.Environment as 'functional' class: step(state, action) -> new_state
        * study in detail the behavior of MCTS while exploring and expanding new nodes
        * rethink stores of the form store[state_int] = [val_action1, val_action2, ....] 
        * if stores are reformed (currently in works) => how to deal with Min-Max ?
