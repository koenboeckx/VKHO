
# Summary

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
    * First: IQL with DQN for each agent of player 1
    * First: IQL with DQN for each agent of player 1
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
        * study in detail the behavior of MCTS while exploring and expanding new nodes: **ok**
        * rethink stores of the form store[state_int] = [val_action1, val_action2, ....]. *For now*: back to single stores store[state] = value
        * **Change**: 'State' is now immutable => can be used as hash key. Also: include player to play in State.
        * Division by Zero in UCB: consequence of reintitialising node after identified as a child node (effectively: reset of .n_visits[state] = 0). *Solution*: check if node is already present in store before (re)setting initial values.
        * if stores are reformed (currently in works) => how to deal with Min-Max ?: *solution*: this doesn't solve the core problem (see next point). However, alternatively updating v_values in function of wich player (`v_values[state] += or -= reward`, accordingly)
        * How to handle loops in behavior (returning to same state after X actions) -> leads to loop where we never find a leaf node = **core of the problem**
        * Added to MCTS: save and load methods that store/recover the 3 essential stores of the object: .n_visits, .v_values, .children
        * Remark: being able to return previous states introduces loops that are not present in other board games like 'Go' (but are in chess)
    1. [22Oct19] Rewrite to store everything in separate 'stores'
    1. [24Oct19] Implemented IQL:
        * Use DQN with experience replay and target network for both agents
        * Implement first naively, then with updates over combined minibatch loss
    1. [03Nov19] Intermediate corrections
        * Correction: addition of 'state = next_state'
        * Addition of agent.optim.zero_grad()
        * Rewrite of loss calculation
        * Addition of next_values_v = next_values_v.detach() # !! avoids feeding gradients in target network
        * Correction of done_mask
        * Expansion of epsilon calculation
        * Experience namedtuple
        * Simplification of classification layer
    1. [04Nov19] After 1M step simulation, evaluate and find the errors in the two created model files.
    1. [04Nov19] Work on TAN game (Independent Q-Learning article)
    1. [09Nov19] Corrections to game:
        * included Linr-Of-Sight criterion for aiming UPDATE: done
        * Working with window around future position to avoid agents coming too close UPDATE: done
    1. [10Nov19] Began development of Policy Gradients
    1. [11Nov19] IDEA: implement a wrapper for a AI Gym Env (like CartPole) to (1) allow same interface as game Environment, and (2) allow for easy (and secure) experimenting for a single agent before moving on to multiple agents in the battlefield setting. => [18Nov19] **DONE** works with actor-critic2
    1. [11Nov19] PG (in REINFORCE) works (even two simultaneous learners) for small board (5x5) against RANDOM OPPONENT without additional measures (i.e. actor)-> To improve for more complex settings. Doesn't work on 11x11 board (even with extended state repr for net) but [14Nov19] works on 7x7 and results have been documented.
    1. [11Nov19]: First priority: improve state representation for DNN to include e.g. ammo: DONE [12Nov19]
    1. [12Nov19]: IDEA: work iteratively: learn policy against random agent; then retrain agents against agent's using this policy => Q: how to adapt policy/model so it can be used by opponent?
    1. [12Nov19]: implement critic and extend REINFORCE => [14Nov19]: actor-critic is partially done, but contains error; TODO: solve this
    1. [14Nov19]: TODO: implement experiment manager like 'sacred' 
    1. [18Nov19]: open ai CartPole-v0 single agent works (well) with actor-critic2; however, this doesn't work with battlefield game => is this an error or is more going on? **To investigate**
    1. [19Nov19]: removed SummaryWriter; all experiments are defined with `sacred` and stored in a MongoDB.
    1. [26Nov19]: Test simple implementation of DQN and tabular Q-learning on Cartpole (discretized if needed)
    1. [27Nov19]: Began development of GRU agent model
    1. [07Dec19]: TODO: use variable identifiers in parameter lists
    1. [02Jan20]: Idea: add "Action-Filter" as in "Skynet: A Top Deep RL Agent in the Inaugural Pommerman Team Competition" to improve learning speed by telling the agent initially which actions not to take.
    1. [02Jan20]: Apply 'curriculum learning', where agents are trained against progressivley stronger opponents (see Bengio et al, Curriculum Learning, 2009):
        1. `StaticAgentsAgent`
        1. `RandomNoFireAgent`
        1. `RandomFireAgent`
        1. `PreviouslyTrainedAgent`
    1. [14Jan20]: important: NORMALIZE inputs => e.g. ammo: if used on face value (e.g. 500) -> leads initially to large logits, then to prob dist centered on a single value and hence:
        1. no exploration
        1. `ln pi(a|s)` very close to zero -> small gradients
    TODO: test all this on newly initiated network to confirm hypothesis.
    1. [16Jan20]: can we compute variance of gradient with and without baseline?
        * [20Jan20] UPDATE: an attempt was made -> results are non-conclusive
    1. [17Jan20]: write infrastructure to transfer learning from own troops to enemies
        * [20Jan20] UPDATE: implemented correction so that all agents can be trained (both own agents and opponents)
    1. [20Jan20]: write routine to capture statistiscs (e.g. how many times `fire` after `aim`)
    1. [21Jan20]: Can agent choose and execute action when dead, i.e. does it matter if we use `actions = [agent.get_action(state) if agent.alive else 0 for agent in agents]` or is `actions = [agent.get_action(state) for agent in agents]` correct? -> this goes to exploration. Anyhow, the former way doesn't work => not unlogical, since agent has no way to explore other actions than `do_nothing` if `state[agent].alive == False`
        * [21Jan20]: (TODO) Idea: explore how NN works when own agent alive flag is set to 0 (e.g. quid generated logits?)
    1.         
    