import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from arguments import get_args
from utils import storage, setup_seed
from plot import plot
from env import twoColorsEnv
from Rollout import Rollout
from Policy import ActorCritic
from agent import A2C

def train(env, rollouts, rew_step, agent, model, max_steps, gamma, device):
    reward_trace = 0
    cumulative_rewards = []

    state = env.reset()
    rollouts.states[0].copy_(torch.FloatTensor(state))
    rollouts.to(device)
    
    for step in tqdm(range(max_steps)):
     
        with torch.no_grad():
            action, value, log_prob = model(rollouts.states[rollouts.step])
        
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        next_state = torch.FloatTensor(next_state)

        rollouts.store(next_state, action, reward, value, log_prob)
        
        reward_trace += reward
        cumulative_rewards.append(reward_trace)

        rew_step.insert(cumulative_rewards)
  
        if rollouts.step == 0:   
            with torch.no_grad():    
                _, next_value, _ = model(rollouts.states[-1])

            rollouts.values[-1].copy_(next_value)

            rollouts.compute_returns(gamma)
            actor_loss, critic_loss, entropy = agent.update(rollouts)
            rollouts.after_update()
            
    return cumulative_rewards, rew_step.get()

def main():
    args = get_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    seeds = random.sample(range(args.seed, args.seed + args.num_seeds * 1000), args.num_seeds) 
    steps_per_rollout = 16
    hidden_size = 256
    shade_flag = True if args.num_seeds > 1 else False

    _rew_step           = []
    _cumulative_rewards = []

    print(f"Total {args.num_seeds} seeds to run:")
    for i in range(args.num_seeds):
        print(f"seed {i} begins: ")
        setup_seed(seeds[i])
        env = twoColorsEnv()

        model = ActorCritic(env.observation_shape, env.action_space.n, hidden_size).to(device)
        rollouts = Rollout(steps_per_rollout, env.observation_shape, env.action_space.n, device=device)
        rew_step = storage(args.max_steps)

        if args.optim == "SGD":
            agent = A2C(model, args.epsilon_EN, lr=args.lr)
        elif args.optim == "Adam":
            agent = A2C(model, args.epsilon_EN, lr=args.lr, sgd=False)

        cumulative_rewards, rew_step = train(env, rollouts, rew_step, agent, model, args.max_steps, args.gamma, device)

        _rew_step.append(rew_step)
        _cumulative_rewards.append(cumulative_rewards)

    a = np.array(_cumulative_rewards)
    b = np.array(_rew_step)
    s1 = np.average(a, axis=0)
    s2 = np.average(b, axis=0)

    if shade_flag:
        s1_std = a.std(axis=0)
        s2_std = b.std(axis=0)
        plot(args.max_steps, s1, s2, s1_std, s2_std, shade_flag=shade_flag)
    else:
        plot(args.max_steps, s1, s2)


if __name__ == "__main__":
    main()