from common.plot import plot
from common.arguments import get_args

args = get_args()

plot(int(args.max_steps),
     args.log_dir,
     args.radius,
     args.log_interval,
     args.steps_per_rollout,
     args.algo)

print("plot successfully!")
