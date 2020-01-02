import enum
import gym
import numpy as np
from lib import data

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class Actions(enum.Enum):
    Skip  = 0
    Buy   = 1
    Close = 2

class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        "Alternative instance creator"
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)
    
    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT, commission=DEFAULT_COMMISSION_PERC,
                    reset_on_close=True, state_1d=False, random_ofs_on_reset=True, reward_on_close=False,
                    volumes=False):
        """
        :param prices:  contains one or more stockprices for one or more instruments as a dict,
                        with keys the intrument's name and values is a container object data.Prices
                        which holds price data arrays
        :param bars_count:  the count of bars we pass in observation (default=10)                        
        :param commission:  percentage of stock price to pay to broker for transaction (default=0.1)
        :param state_1d:    switches between two represesentations
        :param random_ofs_on_reset: start at random time position after each env reset
        :param reward_on_close: switches between reward schemes
        :param volumes:     include volumes in observations
        """
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close,
                                  reward_on_close=reward_on_close, volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close,
                                  reward_on_close=reward_on_close, volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape,
                                                dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset                                               
        self._seed()
    
    def reset(self):
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()
    
    def step(self, action_idx):
        """wrapper around the state.mtethods where the real action happens."""
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {'instrument': self._instrument,
                'offset': self._state._offset}
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        pass 

    def close(self):
        "gets called on destruction of environment to free resources"
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]
    
class State:
    def __init__(self, bars_count, commission_perc, reset_on_close,
                 reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
    
    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset
    
    @property
    def shape(self):
        """The State class is encoded into a single vector, which includes prices, 
        with optional volumes and two numbers indicating the presence of a bought share
        and position profit."""
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (4 * self.bars_count + 1 + 1, )
        else:
            return (3 * self.bars_count + 1 + 1, )
        
    def encode(self):
        """
        Convert current state into a numpy array.
        """
        res = np.ndarray(shape=self.shape, dtpye=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        return res
    
    def _cur_close(self):
        "Compute current bar's close price"
        open = self._prices.open[self._offset]
        rel_close = self._prices.open[self._offset]
        return open * (1.0 + rel_close)
    
    def step(self, action):
        """Returns: (1) the reward in percentage
                    (2) an indication of the episode pending
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close # assumes instant order execution at current bar's close price
            reward = self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price

            self.have_position = False
            self.open_price = 0.0 

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close
        
        return reward, done 
    
class State1d(State):
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)
    
    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count - 1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        if self.volumes:
            res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / self.open_price
        return res


        