from utilities import LinearScheduler

class Args:
    n_steps =               50000
    board_size =            7
    init_ammo =             5
    max_range =             3
    step_penalty =          0.01
    max_episode_length =    100
    gamma =                 0.99
    n_hidden =              128
    scheduler =             LinearScheduler
    buffer_size =           500
    batch_size =            128
    sync_interval =         90
    lr =                    0.0001
    clip =                  10
    scheduler_steps =       500000
    n_episodes_per_step =   25
    n_enemies =             2
    n_friends =             1
    n_agents =              n_enemies + n_friends + 1

    mixer =                 'QMIX'  # 'VDN' or 'QMIX'
    model =                 'RNN'   # 'FORWARD' or 'RNN'

    path = '/home/koen/Programming/VKHO/new_env/agent_dumps/'

args = Args()