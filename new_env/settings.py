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
    buffer_size =           5000
    batch_size =            512
    sync_interval =         90
    lr =                    0.0001
    clip =                  10
    scheduler_steps =       500000
    n_episodes_per_step =   25
    n_enemies =             3

args = Args()