# Agent57
This repository contains unofficial code reproducing Agent57, which outperformed humans in all Atari games.

# Directory File
1. **agent.py**
    
    define agent to play a supecific environment.

2. **buffer.py**

    define buffer to store experiences with priorites.

3. **learner.py**

    define learner to update parameter such as q networks and functions related to intrinsic reward.

4. **main.py**

    run the main pipeline.

5. **model.py**

    define some models such as q network and functions related to intrinsic reward. 

6. **segment_tree.py**

    define segment tree which decide segment index according to the priority.

7. **tester.py**

    define tester which test performance of Agent57.

8. **utils.py**

    define some classes and functions such as UCB and Retrace operator.


# Requirement
* python==3.9.5

* matplotlib==3.4.2
* ray==1.4.1
* lz4==3.1.3
* numpy==1.21.0
* omegaconf==2.1.1
* torch==1.9.0


# Installation
```bash
pip install -r requirements.txt
```

# Usage
```bash
python main.py
```

# Citation
Agent57: Outperforming the Atari Human Benchmark

https://arxiv.org/abs/2003.13350
