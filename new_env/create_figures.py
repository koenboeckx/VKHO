"""
Used to generate figures needed for thesis
"""

from tools import plot_window
"""
# 1. Generate 4 figures for part 1 of chapter 4 - experiment 4
runs = { 713:    'REINFORCE'}

filenames = ['grad0.png', 'grad1.png']
for keys, filename in zip([['grads0'], ['grads1']], filenames):
    plot_window(runs=runs, keys=keys, filename=filename, window_size=500, show=False)

plot_window(runs=runs, keys=['length'], filename='mean_length', window_size=500, show=False)
plot_window(runs=runs, keys=['reward'], filename='mean_reward', window_size=2000, show=False)
"""
# 2. Compate REINFORCE - IAC - IQL
runs = {
        713:    'REINFORCE',
        714:    'IAC',
        #716:    'IQL',
        733:    'IQL'
}
plot_window(runs=runs, keys=['reward'], filename='compare_reward', window_size=1000, show=True)