import random
from collections import namedtuple

Experience = namedtuple('Experience', field_names = [
    'state', 'actions', 'rewards', 'next_state', 'done', 'observations', 'next_obs', 'unavailable_actions'
])

class ExponentialScheduler:
    def __init__(self, start, stop, decay=0.99):
        self.stop  = stop
        self.decay = decay
        self.value = start
    
    def __call__(self):
        self.value *= self.value * self.decay
        return max(self.value, self.stop)

class LinearScheduler:
    def __init__(self, start, stop, steps=10000):
        self.start = start
        self.stop  = stop
        self.delta = (start - stop) / steps
        self.t = 0
    
    def __call__(self):
        epsilon =  max(self.start - self.t * self.delta, self.stop)
        self.t += 1
        return epsilon

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.content = []
    
    def __len__(self):
        return len(self.content)
    
    def insert(self, item):
        self.content.append(item)
        if len(self) > self.size:
            self.content.pop(0)
    
    def insert_list(self, items):
        for item in items:
            self.insert(item)
    
    def can_sample(self, N):
        return len(self) >= N
    
    def sample(self, N):
        assert self.can_sample(N)
        return random.sample(self.content, N)