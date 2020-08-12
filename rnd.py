import gym
import gym_maze
import time 

import numpy as np
data = np.load('C:/Users/Omar/Desktop/Astek/Reinforcement-Learning-Coverage-Path-Planning/Game/gym-maze/gym_maze/envs/maze_samples/maze2d_10x10.npy')

def random_agent(episodes=5000):
	env = gym.make("maze-random-10x10-plus-v0")
	env.reset()
	env.render()
	for e in range(episodes):
		action = env.action_space.sample()
		state, reward, done, _ = env.step(action)
		env.render()
		print('Reward', reward)
		#time.sleep(0.1)
		if done:
			break

if __name__ == "__main__":
    random_agent()
