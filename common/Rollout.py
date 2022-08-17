import torch

class Rollout(object):
    def __init__(self, steps_per_rollout, state_shape, action_shape, device):
        self.states = torch.zeros(steps_per_rollout + 1, state_shape)
        self.actions = torch.zeros(steps_per_rollout, 1)
        self.rewards = torch.zeros(steps_per_rollout, 1)
        self.values = torch.zeros(steps_per_rollout + 1, 1)
        self.returns = torch.zeros(steps_per_rollout, 1)

        self.log_probs = torch.zeros(steps_per_rollout, 1)
        
        self.steps_per_rollout = steps_per_rollout
        self.step = 0
        self.device = device

    def to(self, device):
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.log_probs = self.log_probs.to(device)

    def compute_returns(self, gamma):
        self.returns.copy_(self.rewards + gamma * self.values[1:])

    def store(self, state, action, reward, value, log_prob):
        self.states[self.step + 1].copy_(state)
        self.actions[self.step].copy_(action)
        self.rewards[self.step].copy_(torch.as_tensor([reward]))
        self.values[self.step].copy_(value)
        self.log_probs[self.step].copy_(log_prob)

        self.step = (self.step + 1) % self.steps_per_rollout

    def after_update(self):
        self.states[0].copy_(self.states[-1])