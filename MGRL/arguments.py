import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Meta Gradient Reinforcement Learning')
    parser.add_argument(
        '-m',
        '--max-steps',
        type=int,
        default=6e6+4e5,
        help='max steps interacting with environment (default: 6.4M)')
    parser.add_argument(
        '-c',
        '--use-cuda',
        action="store_true",
        default=False,
        help='whether use cuda if available')
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=0,
        help='Initial seed for generating random numbers (default: 0)')
    parser.add_argument(
        '-n',
        '--num-seeds',
        type=int,
        default=1,
        help='The number of seeds to run (default: 1)')
    parser.add_argument(
        '-lr',
        '--lr',
        type=float,
        default=1e-1,
        help='Learning rate of inner optimizer (default: 1e-1)')
    parser.add_argument(
        '-mlr',
        '--meta-lr',
        type=float,
        default=1e-4,
        help='Learning rate of outer optimizer (default: 1e-4)')
    parser.add_argument(
        '-em',
        '--epsilon-meta',
        type=float,
        default=0.12,
        help='Entropy regularization of meta object (default: 0.12)')
    parser.add_argument(
        '-g',
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor (default: 0.99)')
    parser.add_argument(
        '-T',
        '--T',
        type=int,
        default=15,
        help='T rollouts (default: 15)')
    parser.add_argument(
        '-opt',
        '--metaopt',
        type=str,
        default="TorchOpt",
        help='Meta-optimizer (choice: ["TorchOpt", "MetaOptim"])')
    args = parser.parse_args()

    return args