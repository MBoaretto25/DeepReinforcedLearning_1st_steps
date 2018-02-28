# -*- coding: utf-8 -*-
"""
From: http://www.pinchofintelligence.com/introduction-openai-gym-part-2-building-deep-q-network/
@author: marco
"""
## Imports
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import random

## Create gym environmnet 
env = gym.make('CartPole-v0')
observation = env.reset()

# Network input : Definition of the placeholders
networkstate = tf.placeholder(tf.float32, [None, 4], name="input")
networkaction = tf.placeholder(tf.int32, [None], name="actioninput")
networkreward = tf.placeholder(tf.float32,[None], name="groundtruth_reward")
action_onehot = tf.one_hot(networkaction, 2, name="actiononehot")

# The variable in our network: Weights and biases 
w1 = tf.Variable(tf.random_normal([4,16], stddev=0.35), name="W1")
w2 = tf.Variable(tf.random_normal([16,32], stddev=0.35), name="W2")
w3 = tf.Variable(tf.random_normal([32,8], stddev=0.35), name="W3")
w4 = tf.Variable(tf.random_normal([8,2], stddev=0.35), name="W4")
b1 = tf.Variable(tf.zeros([16]), name="B1")
b2 = tf.Variable(tf.zeros([32]), name="B2")
b3 = tf.Variable(tf.zeros([8]), name="B3")
b4 = tf.Variable(tf.zeros(2), name="B4")

# The network layout, initialize the layers
layer1 = tf.nn.relu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,w2), b2), name="Result2")
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2,w3), b3), name="Result3")
predictedreward = tf.add(tf.matmul(layer3,w4), b4, name="predictedReward")

## Learning 
# Getting the Qreward which is a vector resultant from the multiplication of the predicted
# rewards and the actions from the minibatch. Resultant(32,2)
qreward = tf.reduce_sum(tf.multiply(predictedreward, action_onehot), reduction_indices = 1)
# Calculate the loss function from the network reward(ground true) and the predicted qreward
loss = tf.reduce_mean(tf.square(networkreward - qreward))
tf.summary.scalar('loss', loss)
optimizer = tf.train.RMSPropOptimizer(0.0001).minimize(loss)
merged_summary = tf.summary.merge_all()

## Var Initialization
sess = tf.InteractiveSession()
summary_writer = tf.summary.FileWriter('trainsummary',sess.graph)
sess.run(tf.global_variables_initializer())

## Replay 
replay_memory = [] # (state, action, reward, terminalstate, state_t+1)
epsilon = 1.0
epsilon_decay = 0.999
BATCH_SIZE = 32
GAMMA = 0.9
MAX_LEN_REPLAY_MEMORY = 30000
FRAMES_TO_PLAY = 30001
MIN_FRAMES_FOR_LEARNING = 1000
summary = None

for i_epoch in range(FRAMES_TO_PLAY):
    
    ### Select an action and perform this
    ### EXERCISE: this is where your network should play and try to come as far as possible!
    ### You have to implement epsilon-annealing yourself

    ## Without epsilon annealing
#    action = env.action_space.sample() 
    
    ### Epsilon annealing
    if  np.random.rand() <= epsilon: #generate a random value
        action = random.randrange(env.action_space.n)
    else: # or let the agent decide which one is best giving the actual state
        pred_reward = sess.run(predictedreward, feed_dict={networkstate:observation.reshape([1,4])})
        action = np.argmax(pred_reward[0])
        
    # Start Decaying the Epsilon in order to perform epsilon annealing
    # Decay the epsilon every 10t frame and until it reaches the minimum boundary
    if (epsilon > 0.00001 and i_epoch % 10 ==0):
        epsilon *= epsilon_decay
    
    #Get the newobservation and reward giving the action from the previous operation
    newobservation, reward, terminal, info = env.step(action)

    ### I prefer that my agent gets 0 reward if it dies
    if terminal: 
        reward = 0 
        
    ### Add the observation to our replay memory
    replay_memory.append((observation, action, reward, terminal, newobservation))
    
    ### Reset the environment if the agent died
    if terminal: 
        newobservation = env.reset()
    observation = newobservation
    
    ### Learn once we have enough frames to start learning
    if len(replay_memory) > MIN_FRAMES_FOR_LEARNING: 
        # From the memory variable, extract the training batch
        experiences = random.sample(replay_memory, BATCH_SIZE)
        totrain = [] # (state, action, delayed_reward)
        
        ### Calculate the predicted reward gibing the previous states from the batch
        nextstates = [var[4] for var in experiences]
        pred_reward = sess.run(predictedreward, feed_dict={networkstate:nextstates})    
            
        ### Set the "ground truth": the value our network has to predict:
        ### Calculate the Q(s,a) value giving the values from the batch.
        for index in range(BATCH_SIZE):
            state, action, reward, terminalstate, newstate = experiences[index]
            predicted_reward = max(pred_reward[index])
            
            if terminalstate:
                delayedreward = reward
            else:
                delayedreward = reward + GAMMA*predicted_reward
            totrain.append((state, action, delayedreward))
            
        ### Feed the train batch to the algorithm 
        states = [var[0] for var in totrain]
        actions = [var[1] for var in totrain]
        rewards = [var[2] for var in totrain]
        ## Calculate the loss function 
        _, l, summary = sess.run([optimizer, loss, merged_summary], feed_dict={networkstate:states, networkaction: actions, networkreward: rewards})


        ### If our memory is too big: remove the first element
        if len(replay_memory) > MAX_LEN_REPLAY_MEMORY:
                replay_memory = replay_memory[1:]

        ### Show the progress 
        if i_epoch%100==1:
            summary_writer.add_summary(summary, i_epoch)
        if i_epoch%1000==1:
            print("Epoch %d, loss: %f, e: %f" % (i_epoch,l,epsilon))
       

### Play till we are dead
observation = env.reset()
term = False
predicted_q = []
frames = []
for _ in range(1000):
    time.sleep(0.1)
    env.render();
    pred_q = sess.run(predictedreward, feed_dict={networkstate:[observation]})
    predicted_q.append(pred_q)
    action = np.argmax(pred_q)
    observation, _, term, _ = env.step(action)

env.close()

### Plot the replay!
plt.plot([var[0] for var in predicted_q])
plt.legend(['left', 'right'])
plt.xlabel("frame")
plt.ylabel('predicted Q(s,a)')

## ALternative to co-op with the bot
#%matplotlib inline
#plt.ion()
#observation = env.reset()
#
#### We predict the reward for the initial state, if we are slightly below this ideal reward, let the human take over. 
#TRESHOLD = max(max(sess.run(predictedreward, feed_dict={networkstate:[observation]})))-0.2
#TIME_DELAY = 0.5 # Seconds between frames 
#terminated = False
#while not terminated:
#    ### Show the current status
#    now = env.render(mode = 'rgb_array')
#    plt.imshow(now)
#    plt.show()
#
#    ### See if our agent thinks it is safe to move on its own
#    pred_reward = sess.run(predictedreward, feed_dict={networkstate:[observation]})
#    maxexpected = max(max(pred_reward))
#    if maxexpected > TRESHOLD: 
#        action = np.argmax(pred_reward)
#        print("Max expected: " + str(maxexpected))
#        time.sleep(TIME_DELAY)
#    else:
#        ### Not safe: let the user select an action!
#        action = -1
#        while action < 0:
#            try:
#                action = int(raw_input("Max expected: " + str(maxexpected) + " left (0) or right(1): "))
#                print("Performing: " + str(action))
#            except:
#                pass
#    
#    ### Perform the action
#    observation, _, terminated, _ = env.step(action)
#
#print("Unfortunately, the agent died...")
