import argparse

def get_args():
    parser = argparse.ArgumentParser(description='a2c, bmg and mgrl')

    parser.add_argument(
        '--algo',
        type=str,
        default="BMG", 
        help='algorithms (choice: [A2C, BMG, MGRL])')
    parser.add_argument(
        '--max-steps',
        type=int,
        default=6e6+4e5,
        help='max steps interacting with environment (default: 6.4M)')
    parser.add_argument(
        '--use-cuda',
        action="store_true",
        default=False,
        help='whether use cuda if available')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Initial seed for generating random numbers (default: 0)')
    parser.add_argument(
        '--num-seeds',
        type=int,
        default=1,
        help='The number of seeds to run (default: 1)')
    parser.add_argument(
        '--epsilon-EN',
        type=float,
        default=3e-1,
        help='Entropy regularization coefficient (default: 3e-1)')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-1,
        help='Learning rate of inner optimizer (default: 1e-1)')
    parser.add_argument(
        '--meta-lr',
        type=float,
        default=1e-4,
        help='Learning rate of outer optimizer (default: 1e-4)')
    parser.add_argument(
        '--epsilon-meta',
        type=float,
        default=0.12,
        help='Entropy regularization of meta object of mgrl (default: 0.12)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor (default: 0.99)')
    parser.add_argument(
        '--K',
        type=int,
        default=7,
        help='K rollouts for bmg(default: 7)')
    parser.add_argument(
        '--L',
        type=int,
        default=7,
        help='L rollouts for bmg (default: 9)')
    parser.add_argument(
        '--T',
        type=int,
        default=15,
        help='T rollouts for mgrl (default: 15)')    
    parser.add_argument(
        '--optim',
        type=str,
        default="SGD", 
        help='optimizer for a2c (choice: [SGD, Adam])')
    parser.add_argument(
        '--radius',
        type=int,
        default="10", 
        help='window size of smoothing curve (default: 10)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default="100",
        help='steps to log data (default: 100)')
    parser.add_argument(
        '--log-dir',
        type=str,
        default="/logs/",
        help='log directory')
    
    args = parser.parse_args()
    
    return args


    args = parser.parse_args()
