import os
import random
import gym
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize
from atari_helper import *

tf.reset_default_graph()

# Control parameters
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 200000          # Number of frames the agent sees between evaluations
EVAL_STEPS = 10000               # Number of frames for one evaluation
NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network. 
                                 # According to Mnih et al. 2015 this is measured in the number of 
                                 # parameter updates (every four actions), however, in the 
                                 # DeepMind code, it is clearly measured in the number
                                 # of actions the agent choses
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions, 
                                 # before the agent starts learning
MAX_FRAMES = 30000000            # Total number of frames the agent sees 
MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an 
                                 # evaluation episode
UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output 
                                 # has the shape (1,1,1024) which is split into two streams. Both 
                                 # the advantage stream and value stream have the shape 
                                 # (1,1,512). This is slightly different from the original 
                                 # implementation but tests I did with the environment Pong 
                                 # have shown that this way the score increases more quickly
LEARNING_RATE = 0.00001          # Set to 0.00025 in Pong for quicker results. 
                                 # Hessel et al. 2017 used 0.0000625
BS = 32                          # Batch size

PATH = "output/"                 # Gifs and checkpoints will be saved here
SUMMARIES = "summaries"          # logdir for tensorboard
RUNID = 'run_1'
os.makedirs(PATH, exist_ok=True)
os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

atari = Atari(ENV_NAME, NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n, 
                                                                atari.env.unwrapped.get_action_meanings()))