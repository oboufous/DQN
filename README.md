# Deep Q Network (DQN)
A Tensorflow implementation of a **Deep Q Network (DQN)** for playing Atari games.
Trained on [OpenAI Gym Atari environments](https://gym.openai.com/envs/#atari).

Compared to the [original method](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), this implementation includes several improvements including:
- A custom-made environment
- More adequate hyper-parameters to the problem
- Smaller network and faster training

## Requirements
Note: Versions stated are the versions I used, however this will still likely work with other versions.

- Ubuntu 16.04 (Atari envs will only work 'out of the box' on Linux)
- python 3.5
- [OpenAI Gym](https://github.com/openai/gym) 0.10.8 (See link for installation instructions + dependencies)
- [tensorflow-gpu](https://www.tensorflow.org/) 1.5.0
- [numpy](http://www.numpy.org/) 1.15.2
- [scipy](http://www.scipy.org/install.html) 1.1.0
- [opencv-python](http://opencv.org/) 3.4.0
- [inotify-tools](https://github.com/rvoicilas/inotify-tools/wiki) 3.14

## Usage
The default environment is 'maze-random-10x10-plus-v0'.
```
  $ python train.py
```
This will train the DQN on the specified environment and periodically save checkpoints to the `/ckpts` folder.

```
  $ ./run_every_new_ckpt.sh
```
This shell script should be run alongside the training script, allowing to periodically test the latest network as it trains. This script will monitor the `/ckpts` folder and run the `test.py` script on the latest checkpoint every time a new checkpoint is saved.

```
  $ python play.py
```
Once we have a trained network, we can visualise its performance in the game environment by running `play.py`. This will play the game on screen using the trained network.


## License
MIT License
