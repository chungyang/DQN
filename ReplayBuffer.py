from collections import deque
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, weight): # added weight to transition
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done, weight))

    def sample(self, batch_size):
        #state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        weights = zip(*self.buffer)[5] #probabilities
        state, action, reward, next_state, done, weight = np.random.choice(len(self.buffer),batch_size,p=weights) # weighted sampling
        return np.concatenate(state), action, reward, np.concatenate(next_state), done, weight

    def __len__(self):
        return len(self.buffer)
