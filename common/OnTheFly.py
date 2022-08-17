import torch
import torchopt
from .utils import storage
from collections import deque
from torch.distributions.kl import kl_divergence

class OnTheFly():
    def __init__(self, env, steps_per_rollout, max_steps, monitor, device, gamma):
        self.env                = env
        self.reward_trace       = 0                 # trace of reward
        self.cumulative_rewards = deque(maxlen=101) # cumulative rewards
        self.gamma              = gamma
        self.steps_per_rollout  = steps_per_rollout
        self.average_reward     = [0] * 10
        self.monitor            = monitor        
        self.device             = device

    def rollout(self, inner_model, outer_model, TB_flag=False):
        log_probs = []
        values    = []
        rewards   = []
        rollout_reward = 0
        entropies = 0
        state = self.env.reset()
        done = False
        if TB_flag:
            states = []

        # Rollout a trajectory with 16 samples
        for step in range(self.steps_per_rollout):
            state = torch.FloatTensor(state).to(self.device)
            value, action, log_prob, entropy = inner_model(state)

            if TB_flag:
                states.append(state)

            next_state, reward, done, _ = self.env.step(action.cpu().numpy())

            self.reward_trace += reward # for calculating cumulative reward and moving average reward-to-go

            entropies += entropy
        
            log_probs.append(log_prob.unsqueeze(0).to(self.device))
            values.append(value)
            rewards.append(torch.tensor(reward).unsqueeze(0).to(self.device))
            rollout_reward += reward
            
            state = next_state

            self.cumulative_rewards.append(self.reward_trace)
            self.monitor.insert(self.cumulative_rewards)

        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value, _, _, _ = inner_model(next_state)
        values.append(next_value)

        rewards = torch.cat(rewards)
        log_probs = torch.cat(log_probs)
        values    = torch.cat(values)

        self.average_reward[:-1] = self.average_reward[1:] # one left shift  
        self.average_reward[-1] = rollout_reward / self.steps_per_rollout # assign new value
        epsilon_EN = outer_model(torch.FloatTensor(self.average_reward).to(self.device))

        self.monitor.insert_entropy_rate(float(epsilon_EN))
        
        returns = rewards + self.gamma * values[1:]
        returns = returns.detach()
        advantage = returns - values[:-1]

        # Objects to be updated
        actor_loss   = -(log_probs * advantage.detach()).mean()
        critic_loss  = 0.5 * advantage.pow(2).mean()
        entropy_loss = -entropies / self.steps_per_rollout

        # Terms of policy gradient, temporal difference and entropy 
        if TB_flag:
            loss = actor_loss
            return loss, states
        else:
            loss = actor_loss + critic_loss + epsilon_EN * entropy_loss
            return loss, actor_loss, entropy_loss

def matching_function(Kth_model, TB_model, states, Kth_state_dict):
    states = torch.vstack(states)

    torchopt.recover_state_dict(Kth_model, Kth_state_dict)
    dist_K = Kth_model.get_dist(states) # get Kth rollout distribution \pi_\theta{K} before being recovered
    with torch.no_grad():
        dist_T = TB_model.get_dist(states)

    # Calculate KL divergence loss
    KL_loss = kl_divergence(dist_T, dist_K).sum()

    return KL_loss
