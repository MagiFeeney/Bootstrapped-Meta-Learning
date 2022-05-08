import torch
import TorchOpt
from utils import storage
from torch.distributions.kl import kl_divergence

class OnTheFly():
    def __init__(self, env, steps_per_rollout, max_steps, device, discount_factor):
        self.env                = env
        self.reward_trace       = 0  # trace of reward
        self.cumulative_rewards = [] # cumulative rewards
        self.entropy_rate        = [] # track entropy regularization rate of each rollout
        self.gamma              = discount_factor
        self.steps_per_rollout  = steps_per_rollout
        self.average_reward     = [0] * 10
        self.rew_step           = storage(max_steps)
        self.device             = device

    def rollout(self, inner_model, outer_model):
        log_probs = []
        values    = []
        rewards   = []
        rollout_reward = 0
        entropy = 0
        state = self.env.reset()
        done = False

        # Rollout a trajectory with 16 samples
        for step in range(self.steps_per_rollout):
            state = torch.FloatTensor(state).to(self.device)
            dist, value = inner_model(state)

            action = dist.sample()

            next_state, reward, done, _ = self.env.step(action.cpu().numpy())

            self.reward_trace += reward # for calculating cumulative reward and moving average reward-to-go

            log_prob = dist.log_prob(action)
            entropy += -dist.entropy()
        
            log_probs.append(log_prob.unsqueeze(0).to(self.device))
            values.append(value)
            rewards.append(torch.tensor(reward).unsqueeze(0).to(self.device))
            rollout_reward += reward
            
            state = next_state

            self.cumulative_rewards.append(self.reward_trace)
            self.rew_step.insert(self.cumulative_rewards)

        next_state = torch.FloatTensor(next_state).to(self.device)
        _, next_value = inner_model(next_state)
        values.append(next_value)

        rewards = torch.cat(rewards)
        log_probs = torch.cat(log_probs)
        values    = torch.cat(values)
        entropy = entropy / self.steps_per_rollout

        self.average_reward[:-1] = self.average_reward[1:] # one left shift  
        self.average_reward[-1] = rollout_reward / self.steps_per_rollout # assign new value
        epsilon_EN = outer_model(torch.FloatTensor(self.average_reward).to(self.device))

        self.entropy_rate.append(float(epsilon_EN)) 
        
        returns = rewards + self.gamma * values[1:]
        returns = returns.detach()
        advantage = returns - values[:-1]

        # Objects to be updated
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        # Terms of policy gradient, temporal difference and entropy 
        loss = actor_loss + critic_loss + epsilon_EN * entropy
        return loss, actor_loss, entropy