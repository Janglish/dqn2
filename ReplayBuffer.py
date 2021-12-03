import zlib
import pickle
import random
import numpy as np
from collections import namedtuple

class ReplayBuffer:
    
    def __init__(self, capacity, Transition):
        self.Transition = Transition
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, args, Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        experience = self.Transition(*args)
        experience = zlib.compress(pickle.dumps(experience))
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(np.arange(len(self.buffer)), replace=False, size=batch_size)
        experiences = tuple(pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices)
        return experiences
    
    def __len__(self):
        return len(self.buffer)