# -*- coding: utf-8 -*-
"""
Based on : http://www.pinchofintelligence.com/introduction-openai-gym-part-2-building-deep-q-network/
@author: marco
"""
## Imports
import numpy as np
import gym


# Create the environment and display the initial state
env = gym.make('CartPole-v0')

# Create a Simulation of a small run with 200 steps
def run_episode(env, parameters):  
    """Runs the env for a certain amount of steps with the given parameters. Returns the reward obtained"""
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1 # Select Actions given Paramams.observ                                    
        observation, reward, done, info = env.step(action)  #update env with step
        totalreward += reward
        if done:
            break
    return totalreward

# Random search: try random parameters between -1 and 1, see how long the game lasts with those parameters
bestparams = None  
bestreward = 0  
for _ in range(10000):  
    parameters = np.random.rand(4) * 2 - 1 #randomly generate an array with the same size of the observations
    reward = run_episode(env,parameters)
    if reward > bestreward:    #if the reward is higher the the current bestreward
        bestreward = reward    # update bestreward
        bestparams = parameters #save the parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break
            
# For animation purposes, it is actually the same as the first simulation, however in this one
# you can actually see the cartpole move given the best parameters from the random search.
            
def show_episode(env, parameters):  
    """ Records the frames of the environment obtained using the given parameters... Returns RGB frames"""
    observation = env.reset()
    firstframe = env.render(mode = 'rgb_array')
    frames = [firstframe]
    
    for _ in range(5000):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        print(action)
        print(parameters)
        observation, reward, done, info = env.step(action)
        frame = env.render(mode = 'rgb_array')
        frames.append(frame)
        if done:
            print('Done')
            break
    return frames

#performe the simulation
frames = show_episode(env, bestparams)
# Close the environment
env.close()
