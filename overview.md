
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
    1. [24Jan20]: A2C: if no entropy loss -> collapse of pi(a|s) around a single action !!
    1. [24Jan20]: tuning A2C training is too difficult - or something is wrong in my code => DECISION: keep working with REINFORCE for now; revisit A2C later. Next step: replace model with GRU and (implicetly) condition on state trajectory \tau.
        * is there a link with the number of available actions?
    1. [29Jan20]: GRU (re)implemented:
        * comparable (certainly not better) results vs reference REINFORCE
        * no improvement for A2C -> still doesn't work
        * Next: implement COMA
    1. [29Jan20]: Before COMA: reinmplement IAC where update of pi(a|s) if sufficiently slower than the update of Q(.,.):
        * seperate actor and critic
        * update critic every episode; update actor every N episodes
        * From Konda & Tsitsiklis (2000): "Note that the last assumption requires that the actor parameters be updated at a time scale slower than that of critic." 
        * UPDATE: no improvement [29Jan20]
    1. IDEA (based on pymarl): mask out unavailable actions (e.g. no fire when not aiming)
        * requires change to environment to return unavailable actions
        * UPDATE: this won't work with PG, because if actions are never taken, their value is never adjusted in the conditional policy. But can work in Q-learning (?). => []
    1. Remark on negative reward for step -> if the agent can't lose (e.g. only static agents), this will induce the agent to shorten episodes (through emptying ammo), without winning himself. This can also be the case if he can lose (e.g. if step_reward = -.2, it is better to play 5 steps and then lose (total reward 5*-2 - 1 = -2), in stead of playing 20 steps and then win (total reward = 20*-.2 + 1 = -3). (This (agent choosing to end episode) is no longer possible if ammo is infinite.)
    1. TODO: re-implemented so that observation is window centered on agent
        * allows weight sharing across agents (only observation will be different)
        * enables "easy" integration of QMix (use Qtot in stead of Qa)
    1. [12Feb20]: [After complete rewrite of both Environment (-> SimpleEnvironment) and the QL algortihm] (I)QL: removing unavailable actions during both (1) action selection as (2) updating the network weights seems to remove all problems with convergence !!
    1. [15Feb20]: Both IQL and PG work on new environment in 2v2 mode; furthermore, because dsitinction between observation and state and observation written in such a way that it is agent/team independent, transfer of (the single) nn.Module model works too (at least for PG, to test for IQL). Next steps:
        1. either implement IAC, or
        1. implement Q-Mixing (preferably?), or
        1. extend environment to NvN games : DONE [17Feb20]
    1. [16Feb20]: noticed: PG doesn't converge if gamma = 0.9, better: 0.99; IQL works better with gamma=0.9
    1. [19Feb20] Update since 05Feb20 (last meeting with Steven):
        * rewrite observation (agent-centered) -> no improvement; 
        * complete rewrite of the Environment:
            * assignment of team to agents
            * allows multiple agents of both teams
            * disctinction between `Observation` and  `State`:
                - observation specific for each agent
                - contains own position, own ammo, own aiming
                - contains list with relative position of friends
                - contains list with relative position of enemies
            * multiple agents => action space depends on number of enemies
                - is computed during execution
        * `Observation` is processed to tensor as input for network
        * IQL and PG can cope with actions that are not allowed:
            * in action selection
            * in update action
        * Both IQL and PG agents share network weights (*weight sharing*)
        * Implemented RNN with `GRUCell` 
            * changes to:
                * `generate_episode` and `Experience`: keep track of hidden state
                * `update` method for IQL and PG
            * apparent convergence problem -> quid learning rate impact?
            
        * [19Feb20] First implementation of QMIX (with VDN)
    1. [20Feb20] - After issues with low yield after update of env: cause seems to be random initial positions; behavior is much more stable when initial positions are fixed. Furthermore, `max_range` has significant impact on convergence speed. To improve development iteration, use:
        * `board_size` = 7
        * `init_ammo` = 5
        * `max_range` = 5
        * (fixed initial positions)
    1. [22Feb20] TODO: to evaluate (I)QL: add possibility to set eps=0.0 in action selection so that always best action can be chosen during evaluation (<-> training)
    1. [22Feb20]: TODO: MongoDB: how extract info - make better figures