import os, gym, ptan, argparse
import numpy as np 
import torch
import torch.optim as optim

from lib import environ, data, models, common
from tensorboardX import SummaryWriter

BATCH_SIZE = 32
BARS_COUNT = 10
TARGET_NET_SYNC = 1000
DEFAULT_STOCKS = "data/YNDX_160101_161231.csv"

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

REWARD_STEPS = 2

LEARNING_RATE = 0.0001

EPSILON_START = 1.0
EPSILON_STOP  = 0.1
EPSILON_STEPS = 1000000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--data", default=DEFAULT_STOCKS, help="Stocks file or dir to train on, default=" + DEFAULT_STOCKS)
    parser.add_argument("--year", type=int, help="Year to be used for training, if specified, overrides --data option")
    parser.add_argument("--valdata", default=DEFAULT_VAL_STOCKS, help="Stocks data for validation, default=" + DEFAULT_VAL_STOCKS)
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", args.run)
    os.makedirs(saves_path, exist_ok=True)

    if args.year is not None or os.path.isfile(args.data):
        if args.year is not None:
            stock_data = data.load_year_data(args.year)
        else:
            stock_data = {"YNDX": data.load_relative(args.data)}
        env = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=False)
        env_tst = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
    elif os.path.isdir(args.data):
        env = environ.StocksEnv.from_dir(args.data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
        env_tst = environ.StocksEnv.from_dir(args.data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
    else:
        raise RuntimeError("No data to train on")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    writer = SummaryWriter(comment='-simple-' + args.run)
    net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # main training loop
    step_idx = 0
    
    with common.RewardTracker(writer, np.inf) as reward_tracker:
        while True:
            step_idx += 1
            buffer.populate(1)
            selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)