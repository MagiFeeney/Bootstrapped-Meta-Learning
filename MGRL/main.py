import gym
import random
import numpy as np
import TorchOpt
import meta_optim as MetaOptim

import torch
import torch.optim as optim
from env import twoColorsEnv
from Policy import ActorCritic
from Meta import MLP
from copy import deepcopy
from arguments import get_args
from utils import setup_seed
from plot import plot
from OnTheFly import OnTheFly
from Trainer import Trainer

def main():
    args = get_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    seeds = random.sample(range(args.seed, args.seed + args.num_seeds * 1000), args.num_seeds) 

    steps_per_rollout = 16
    hidden_size = 256
    input_size   = 10
    feature_size = 32
    shade_flag = True if args.num_seeds > 1 else False

    _rew_step           = []
    _cumulative_rewards = []
    _entropy_rate       = []

    outer_iters = args.max_steps // steps_per_rollout // args.T

    print(f"Total {args.num_seeds} seeds to run:")
    for i in range(args.num_seeds):
        print(f"seed {i} begins: ")
        setup_seed(seeds[i])
        env = twoColorsEnv()

        inner_model = ActorCritic(env.observation_shape, env.action_space.n, hidden_size).to(device)
        outer_model = MLP(input_size, feature_size).to(device)

        if args.metaopt == "TorchOpt":
            inner_optimizer = TorchOpt.MetaSGD(inner_model, lr=args.lr)
        elif args.metaopt == "MetaOptim":
            inner_optimizer = MetaOptim.MetaSGD([inner_model], lr=args.lr)
        else:
            raise ValueError("Invalid Meta-Opimizer")

        outer_optimizer = optim.Adam(outer_model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), eps=1e-4)

        controller = OnTheFly(env, steps_per_rollout, \
                              outer_iters * steps_per_rollout * args.T, \
                              device, args.gamma)
        trainer    = Trainer(env, controller, inner_model, outer_model, \
                             inner_optimizer, outer_optimizer, outer_iters, \
                             args.epsilon_meta, args.T)

        cumulative_rewards, rew_step, entropy_rate = trainer.train(args.metaopt) 
        
        _rew_step.append(rew_step)
        _cumulative_rewards.append(cumulative_rewards)
        _entropy_rate.append(entropy_rate)

    a = np.array(_cumulative_rewards)
    b = np.array(_rew_step)
    c = np.array(_entropy_rate)
    s1 = np.average(a, axis=0)
    s2 = np.average(b, axis=0)
    s3 = np.average(c, axis=0)

    if shade_flag:
        s1_std = a.std(axis=0)
        s2_std = b.std(axis=0)
        s3_std = c.std(axis=0)
        plot(outer_iters * steps_per_rollout * args.T, s1, s2, s3, s1_std, s2_std, s3_std, shade_flag=shade_flag)
    else:
        plot(outer_iters * steps_per_rollout * args.T, s1, s2, s3)

if __name__ == "__main__":
    main()