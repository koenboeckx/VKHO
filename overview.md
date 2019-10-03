1. Implement simple game:
    * symmetric (2 players with each 2 tanks)
    * only 8 possible actions:
        (0) do_nothing
        (1) aim1, i.e. aim at first agent of other team
        (2) aim2, i.e. aim at second agent of other team
        (3) fire -> hit only if target is close enough
        (4) move_up
        (5) move_down
        (6) move_left
        (7) move_right
    * (maybe) provide unlimited ammo
    * no fuel attribute
    * actions are executed simutaneously (a0, a1, a2, a3)
    * implement visulisation tool based on PyGame
2. Implement MCTS to develop non-trivial strategies for both players
    * centralised control with shared observations
    * consider global game state?
    * how to analyze (NE?, ESS?)
3. Deploy MARL algorithms to develop strategies
    * compare with MCTS results
4. Extend game to more realistic situations with
    * richer action space
    * more evolved state / observations (e.g. partial observability?)

99. Other ideas:
    * MCTS: reduce the game to turn-based game:
        * first execute team1's actions: (a0, a1, 0, 0)
        * evaluate game to see if team 1 won
        * then execute team2's actions:  (0, 0, a2, a3)
        * evaluate game to see if team 2 won
    -> otherwise, joint action space become too large (8**4 = 4096)
    * MCTS: based on FULL state information (thus not only individual observations)
    