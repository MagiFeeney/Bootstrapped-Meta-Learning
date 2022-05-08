## ***BMG***
- **TorchOpt** <br/>
  python3 main.py -m 6400000 -n 1 -lr 1e-1 -mlr 1e-4 -g 0.99 -K 7 -L 9 -opt "TorchOpt"
- **MetaOptim** <br/>
  python3 main.py -m 6400000 -n 1 -lr 1e-1 -mlr 1e-4 -g 0.99 -K 7 -L 9 -opt "MetaOptim"

We use 7 and 9 corresponding to K and L for default, which you can tune if necessary.