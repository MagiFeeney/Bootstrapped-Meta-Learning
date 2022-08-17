import os
import gym
import random
import numpy as np
import torchopt

import torch
import torch.optim as optim
from copy import deepcopy

from common.Policy import ActorCritic
from common.env import twoColorsEnv
from common.arguments import get_args
from common.Meta import MLP
from common.utils import storage, setup_seed
from common.plot import plot
from common.OnTheFly import OnTheFly
from common.Rollout import Rollout
import algo

def main():
    args = get_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    seeds = random.sample(range(args.seed, args.seed + args.num_seeds * 1000), args.num_seeds) 

    steps_per_rollout = 16
    hidden_size = 256
    input_size   = 10

    if args.algo == "BMG" or args.algo == "MGRL":
        feature_size = 32
        outer_iters = int(args.max_steps) // steps_per_rollout // (args.K + args.L)
        
        
    print(f"Total {args.num_seeds} seeds to run:")
    for i in range(args.num_seeds):
        print(f"seed {i} begins: ")
        setup_seed(seeds[i])
        env = twoColorsEnv()

        log_dir = args.log_dir + '/seed-' + str(i) + '/'
        os.makedirs(log_dir, exist_ok=True)
        
        monitor = storage(args.max_steps, args.log_interval, log_dir)
        
        if args.algo == "A2C":
            model = ActorCritic(env.observation_space.shape[0], env.action_space.n, hidden_size).to(device)
            rollouts = Rollout(steps_per_rollout, env.observation_space.shape[0], env.action_space.n, device=device)

            if args.optim == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
            elif args.optim == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=args.lr)

            agent = algo.A2C(env, rollouts, model, optimizer,
                             args.epsilon_EN, monitor, args.max_steps,
                             args.gamma, device)
            
            agent.train()
            
        else:
            inner_model = ActorCritic(env.observation_space.shape[0], env.action_space.n, hidden_size).to(device)
            outer_model = MLP(input_size, feature_size).to(device)

            inner_optimizer = torchopt.MetaSGD(inner_model, lr=args.lr)
            outer_optimizer = optim.Adam(outer_model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), eps=1e-4)

            controller = OnTheFly(env, steps_per_rollout, \
                                  outer_iters * steps_per_rollout * (args.K + args.L), \
                                  monitor, device, args.gamma)

            if args.algo == "BMG":
                agent = algo.BMG(env, controller, inner_model, outer_model, \
                                 inner_optimizer, outer_optimizer, \
                                 outer_iters, args.K, args.L)
                Kth_model = deepcopy(inner_model)
                agent.train(Kth_model)
                
            elif args.algo == "MGRL":
                agent  = algo.MGRL(env, controller, inner_model, outer_model, \
                                   inner_optimizer, outer_optimizer, outer_iters, \
                                   args.epsilon_meta, args.T)
                agent.train()
            else:
                raise ValueError

        monitor.close()

    
    plot(int(args.max_steps),
         args.log_dir,
         args.num_seeds,
         args.radius,
         args.log_interval,
         steps_per_rollout,
         flag=False if args.algo == "A2C" else True)

    print("plot successfully!")
        

if __name__ == "__main__":
    main()
