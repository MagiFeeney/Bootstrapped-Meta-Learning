import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, state):
        value = self.critic(state)
        probs = self.actor(state)
        dist  = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return value, action, log_prob, entropy

    def evaluate(self, states, actions):
        values = self.critic(states)
        probs = self.actor(states)
        dists  = Categorical(probs)
        log_probs = dists.log_prob(actions.flatten()).view(-1, 1)
        entropy   = dists.entropy().mean()
        return values, log_probs, entropy    

    def get_dist(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)

        return dist
