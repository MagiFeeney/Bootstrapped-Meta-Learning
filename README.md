# Bootstrapped Meta-Learning Replication
## Display

| **A2C Baseline** | **Meta-Gradient Reinforcement Learning** | **Bootstrapped Meta-Learning** |
|-------------------|------------------------------------------|--------------------------------|
| <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/raw/main/results/a2c/shaded-cumulative-rewards.png" width="700"/> | <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/raw/main/results/mgrl/cumulative-rewards.png" width="700"/> | <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/raw/main/results/bmg/cumulative-rewards.png" width="700"/> |
| <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/raw/main/results/a2c/shaded-rew-step-1.png" width="700"/> | <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/raw/main/results/mgrl/rew-step.png" width="700"/> | <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/raw/main/results/bmg/rew-step.png" width="700"/> |
|                   | <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/raw/main/results/mgrl/entropy-rate.png" width="700"/> | <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/raw/main/results/bmg/entropy-rate.png" width="700"/> |

## How To Use
### A2C
```
$ python main.py --algo A2C --seed 0 --epsilon-EN 3e-1 --lr 1e-1 --gamma 0.99 --log-dir "logs" --max-steps 6400000
```

Both SGD and Adam are supported, however, it is not encouraged to use Adam otherwise you won't see desired result.

### MGRL
```
$ python main.py --algo MGRL --seed 0 --lr 1e-1 --meta-lr 1e-4 -epsilon-meta 0.12 --gamma 0.99 --T 15 --max-steps 6400000
```

### BMG
```
$ python main.py --algo BMG -seed 0 --lr 1e-1 --meta-lr 1e-4 --gamma 0.99 --K 7 --L 9 --max-steps 6400000
```

We use 7 and 9 corresponding to K and L for default, for details, you can redirect to the original paper [Bootstrapped Meta-Learning](https://arxiv.org/pdf/2109.04504.pdf)

## Visualization
```
python plot.py --algo BMG
```
The results will be stored in `results` folder by default. To plot, please specify algorithm name that corresponds to the latest run.

## Requirements
```
$ pip install -r requirements.txt
```