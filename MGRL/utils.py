import torch
import numpy as np

class storage():
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.rew_step  = []
        self.flag      = True if max_steps > 4e5 else False
        self.step      = 1

    def insert(self, cumulative_rewards):
        if self.flag:
            if self.step >= self.max_steps - 4e5 + 100:
                self.rew_step.append((cumulative_rewards[-1] - cumulative_rewards[-101]) / 100)
                if self.step ==  self.max_steps:
                    for i in reversed(range(99, 0, -1)):
                        self.rew_step.append((cumulative_rewards[-1] - cumulative_rewards[-1 - i]) / i)
        else:
            if self.step == 100:
                self.rew_step.append(cumulative_rewards[-1] / 100)
            elif self.step >= 101:
                self.rew_step.append((cumulative_rewards[-1] - cumulative_rewards[-101])/ 100)
                if self.step ==  self.max_steps:
                    for i in reversed(range(99, 0, -1)):
                        self.rew_step.append((cumulative_rewards[-1] - cumulative_rewards[-1 - i]) / i)

        self.step = self.step + 1

    def get(self):
        return self.rew_step

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True