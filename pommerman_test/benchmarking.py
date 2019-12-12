"An example to show how to set up a pommerman game programmatically"
import pommerman
from pommerman import agents

N_EPISODES = 1000

def main():
    """Simple function to bootstrap a game.

    Use this as an example to set up your training env.
    """
    # Print all possible environements in the Pommerman registry
    # print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    n_wins = 0
    n_ties = 0
    n_loss = 0
    for i_episode in range(N_EPISODES):
        state = env.reset()
        done = False
        while not done:
            #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        #print('Episode {} finished'.format(i_episode))
        if reward[0] == 1:
            n_wins += 1
        elif reward[0] == 0:
            n_ties += 1
        elif reward[0] == -1:
            n_loss += 1
    print('Average wins = {}/{} ({} %)'.format(n_wins, N_EPISODES,
                                                100*n_wins/N_EPISODES))
    print('Average ties = {}/{} ({} %)'.format(n_ties, N_EPISODES,
                                                100*n_ties/N_EPISODES))
    print('Average loss = {}/{} ({} %)'.format(n_loss, N_EPISODES,
                                                100*n_loss/N_EPISODES))                                                                                                
    env.close()

if __name__ == '__main__':
    main()