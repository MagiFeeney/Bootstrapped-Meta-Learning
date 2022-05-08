## ***A2C***
- **SGD** <br/>
  python3 main.py -m 6400000 -s 0 -n 1 -e 0.3 -lr 0.1 -opt "SGD"
- **Adam** <br/>
  python3 main.py -m 6400000 -s 0 -n 1 -e 0.3 -lr 0.1 -opt "Adam"

Both SGD and Adam are supported, however, it is not encouraged to use Adam otherwise you won't see desired result.

Feel free to try more seeds by tuning -n! 