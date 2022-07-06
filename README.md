# Bootstrapped Meta-Learning Replication
## Display

- **A2C Baseline**
     <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/blob/main/A2C/results/shaded-cumulative-rewards.png/" width="700"/>
     <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/blob/main/A2C/results/shaded-rew-step-1.png/" width="700"/>
- **Meta-Gradient Reinforcement Learning**
     <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/blob/main/MGRL/results/TorchOpt/cumulative-rewards.png/" width="700"/>
     <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/blob/main/MGRL/results/TorchOpt/rew-step.png/" width="700"/>
     <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/blob/main/MGRL/results/TorchOpt/entropy-rate.png/" width="700"/>
- **Bootstrapped Meta-Learning**
     <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/blob/main/BMG/results/TorchOpt/cumulative-rewards.png/" width="700"/>
     <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/blob/main/BMG/results/TorchOpt/rew-step.png/" width="700"/>
     <img src="https://github.com/MagiFeeney/Bootstrapped-Meta-Learning/blob/main/BMG/results/TorchOpt/entropy-rate.png/" width="700"/>

## How To Use
### ***A2C***
- **SGD** <br/>
  python3 main.py -m 6400000 -s 0 -n 1 -e 0.3 -lr 0.1 -opt "SGD"
- **Adam** <br/>
  python3 main.py -m 6400000 -s 0 -n 1 -e 0.3 -lr 0.1 -opt "Adam"

Both SGD and Adam are supported, however, it is not encouraged to use Adam otherwise you won't see desired result.

Feel free to try more seeds by tuning -n! 
### ***MGRL***
- **TorchOpt** <br/>
  python3 main.py -m 6400000 -n 1 -lr 1e-1 -mlr 1e-4 -em 0.12 -g 0.99 -T 15 -opt "TorchOpt"
- **MetaOptim** <br/>
  python3 main.py -m 6400000 -n 1 -lr 1e-1 -mlr 1e-4 -em 0.12 -g 0.99 -T 15 -opt "MetaOptim"

### ***BMG***
- **TorchOpt** <br/>
  python3 main.py -m 6400000 -n 1 -lr 1e-1 -mlr 1e-4 -g 0.99 -K 7 -L 9 -opt "TorchOpt"
- **MetaOptim** <br/>
  python3 main.py -m 6400000 -n 1 -lr 1e-1 -mlr 1e-4 -g 0.99 -K 7 -L 9 -opt "MetaOptim"

We use 7 and 9 corresponding to K and L for default, which you can tune if necessary.

## Requirements
* TorchOpt (Meta Optimizer) <br/>
pip install TorchOpt
