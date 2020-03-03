from utilities import LinearScheduler

class Args:
    n_steps =               50000
    board_size =            7
    init_ammo =             5
    max_range =             5 # 3
    step_penalty =          0.01
    max_episode_length =    100
    gamma =                 0.99
    n_hidden =              128
    scheduler =             LinearScheduler
    buffer_size =           500
    batch_size =            128
    sync_interval =         90
    lr =                    0.00005 #0.000001
    beta =                  0.01 #0.05 #0.1 # entroply loss coefficient
    clip =                  10
    scheduler_steps =       100000
    n_episodes_per_step =   25
    n_enemies =             2
    n_friends =             2
    n_agents =              n_enemies + n_friends

    mixer =                 'VDN'      # 'VDN' or 'QMIX' or 'QMIX_NS'
    model =                 'RNN'   # 'FORWARD' or 'RNN'

    path = '/home/koen/Programming/VKHO/new_env/agent_dumps/'
    fixed_init_position =   True

    n_iterations =          5

args = Args()