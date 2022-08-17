import torch
from tqdm import tqdm
from collections import deque

class A2C():
    def __init__ (self, env, rollouts, model, optimizer, epsilon_EN, monitor, max_steps, gamma, device):
        self.env = env
        self.rollouts = rollouts
        self.epsilon_EN = epsilon_EN
        self.optimizer = optimizer
        self.model = model
        self.gamma = gamma
        self.device = device
        self.max_steps = int(max_steps)
        self.monitor = monitor

    def train(self):
        
        reward_trace = 0
        cumulative_rewards = deque(maxlen=101)

        state = self.env.reset()
        self.rollouts.states[0].copy_(torch.FloatTensor(state))
        self.rollouts.to(self.device)

        for step in tqdm(range(self.max_steps)):

            with torch.no_grad():
                value, action, log_prob, entropy = self.model(self.rollouts.states[self.rollouts.step])

            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            next_state = torch.FloatTensor(next_state)

            self.rollouts.store(next_state, action, reward, value, log_prob)

            reward_trace += reward
            cumulative_rewards.append(reward_trace)

            self.monitor.insert(cumulative_rewards)

            if self.rollouts.step == 0:   
                with torch.no_grad():    
                    next_value, _, _, _ = self.model(self.rollouts.states[-1])

                self.rollouts.values[-1].copy_(next_value)

                self.rollouts.compute_returns(self.gamma)
                actor_loss, critic_loss, entropy_loss = self.update()
                self.rollouts.after_update()

    def update(self):

        values, log_probs, entropy = self.model.evaluate(self.rollouts.states[:-1], self.rollouts.actions)
        advantage = self.rollouts.returns - values
        
        actor_loss = -(advantage.detach() * log_probs).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        entropy_loss = -entropy
        
        loss = actor_loss + critic_loss + self.epsilon_EN * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy_loss.item()
