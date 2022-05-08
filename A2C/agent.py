import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class A2C():
    def __init__(self,
                 ActorCritic,
                 epsilon_EN,
                 lr=None,
                 sgd=True):

        self.ActorCritic = ActorCritic
        self.epsilon_EN = epsilon_EN

        if sgd:
            self.optimizer = optim.SGD(ActorCritic.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(ActorCritic.parameters(), lr=lr)

    def update(self, rollouts):

        values, log_probs, entropy = self.ActorCritic.evaluate(rollouts.states[:-1], rollouts.actions)
        advantage = rollouts.returns - values
        
        actor_loss = -(advantage.detach() * log_probs).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        
        loss = actor_loss + critic_loss + self.epsilon_EN * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()