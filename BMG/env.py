from gym import Env 
from gym.spaces import Discrete, Box
import numpy as np
import random

class twoColorsEnv(Env):
    def __init__ (self):
        # Actions we can take, up, down, left, right
        self.grid_length = 5
        self.grid_width = 5
        self.grid_size = self.grid_length * self.grid_width
        try:
            self.targets = random.sample(range(0, self.grid_size), 3)
        except ValueError:
            print('Sample size exceeded population size.')
        self.action_space = Discrete(4)
        self.state = self.embedding()
        self.observation_shape = self.state.shape[0]
        self.rewards = [1., -1.]
        self.flip_count = 100000

    def one_hot_encode(self, x, n_classes):
        return np.eye(n_classes)[x]

    def embedding(self):
        pos = [[self.targets[0] // self.grid_length, (self.targets[0] % self.grid_length)], \
               [self.targets[1] // self.grid_length, (self.targets[1] % self.grid_length)], \
               [self.targets[2] // self.grid_length, (self.targets[2] % self.grid_length)]]
        
        altogether = np.array([])
        for cord in pos:
            altogether = np.append(altogether, self.one_hot_encode(cord, self.grid_length).flatten())
        return altogether

    def rand_loc(self, box):
        if box == 'blue':
            auxiliary = [i for i in range(0, self.grid_size) if i not in self.targets[1:]]
            try:
                self.targets[0] = random.sample(auxiliary, 1)[0]
            except ValueError:
                print('Sample size exceeded population size.')
            
        elif box == 'red':
            auxiliary = [i for i in range(0, self.grid_size) if i not in self.targets[0::2]]
            try:
                self.targets[1] = random.sample(auxiliary, 1)[0]
            except ValueError:
                print('Sample size exceeded population size.')            
        else:
            print('Box To Be Reallocated Not Found')

    def step(self, action):
        action_valid = 1

        # Apply action and observe agents' behavior
        if action == 0:
            position = self.targets[-1] - self.grid_length
            if position >= 0:
                self.targets[-1] = position
        elif action == 1:
            position = self.targets[-1] + self.grid_length
            if position <= self.grid_size - 1:
                self.targets[-1] = position        
        elif action == 2:
            if self.targets[-1] % self.grid_length == 0:
                pass
            else:
                self.targets[-1] -= 1 
        elif action == 3:
            if (self.targets[-1] + 1) % self.grid_length == 0:
                pass
            else:                                                   
                self.targets[-1] += 1 
        else:
            action_valid = 0
            assert action_valid, 'Action invalid!'

        self.flip_count -= 1

        # Calculate reward, observe items' behavior and update state
        if self.targets[-1] == self.targets[0]:
            reward = self.rewards[0]
            self.rand_loc('blue')
            self.state = self.embedding()
            done = True
        elif self.targets[-1] == self.targets[1]:
            reward = self.rewards[1]
            self.rand_loc('red')
            self.state = self.embedding()
            done = True
        else:
            reward = -0.04
            self.state = self.embedding()
            done = False

        # Whether flip is true 
        if self.flip_count == 0:
            self.rewards.reverse()
            self.flip_count = 100000

        info = {}
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        # No pratical meaning for non-episodic environment, just for convention
        return self.state