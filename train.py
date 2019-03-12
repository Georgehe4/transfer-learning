import os
import random
import gym
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize
from atari_helper import *
from replay_memory import *
from environment import *
from dqn import *

# Train on GPU if a trained_path is provided

def train_dqn(main_dqn,  main_dqn_vars, param_summaries, target_dqn=None, target_dqn_vars=[], trained_path = None, save_file = None, model_name="my_model"):
    # Trained path: The path (if provided) to look for the saved file from. EG: "trained/pong/"
    # Save_file: The path (if provided) to save tf outputs to
    # The model name
    """Contains the training and evaluation loops"""
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)   # (★)
    if (target_dqn != None):
        network_updater = TargetNetworkUpdater(main_dqn_vars, target_dqn_vars)
    action_getter = ActionGetter(atari.env.action_space.n, 
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
                                 max_frames=MAX_FRAMES)

    with tf.Session() as sess:
        
        if trained_path != None:
            saver = tf.train.import_meta_graph(trained_path+save_file)
            saver.restore(sess,tf.train.latest_checkpoint(trained_path))
        else:
            sess.run(init)
        
        frame_number = 0
        rewards = []
        loss_list = []
        
        while frame_number < MAX_FRAMES:
            
            ########################
            ####### Training #######
            ########################
            epoch_frame = 0
            while epoch_frame < EVAL_FREQUENCY:
                terminal_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                for _ in range(MAX_EPISODE_LENGTH):
                    # (4★)
                    action = action_getter.get_action(sess, frame_number, atari.state, main_dqn)   
                    # (5★)
                    processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)  
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward
                    
                    # (7★) Store transition in the replay memory
                    my_replay_memory.add_experience(action=action, 
                                                    frame=processed_new_frame[:, :, 0],
                                                    reward=reward, 
                                                    terminal=terminal_life_lost)   
                    
                    if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        loss = learn(sess, my_replay_memory, main_dqn, target_dqn,
                                     BS, gamma = DISCOUNT_FACTOR) # (8★)
                        loss_list.append(loss)
                    if target_dqn != None and frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        network_updater.update_networks(sess) # (9★)
                    
                    if terminal:
                        terminal = False
                        break

                rewards.append(episode_reward_sum)
                
                # Output the progress:
                if len(rewards) % 10 == 0:
                    # Scalar summaries for tensorboard
                    if frame_number > REPLAY_MEMORY_START_SIZE:
                        summ = sess.run(PERFORMANCE_SUMMARIES, 
                                        feed_dict={LOSS_PH:np.mean(loss_list), 
                                                   REWARD_PH:np.mean(rewards[-100:])})
                        
                        SUMM_WRITER.add_summary(summ, frame_number)
                        loss_list = []
                    # Histogramm summaries for tensorboard
                    summ_param = sess.run(param_summaries)
                    SUMM_WRITER.add_summary(summ_param, frame_number)
                    
                    print(len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open('rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number, 
                              np.mean(rewards[-100:]), file=reward_file)
            
            ########################
            ###### Evaluation ######
            ########################
            terminal = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluate_frame_number = 0
            
            for _ in range(EVAL_STEPS):
                if terminal:
                    terminal_life_lost = atari.reset(sess, evaluation=True)
                    episode_reward_sum = 0
                    terminal = False
               
                # Fire (action 1), when a life was lost or the game just started, 
                # so that the agent does not stand around doing nothing. When playing 
                # with other environments, you might want to change this...
                action = 1 if terminal_life_lost else action_getter.get_action(sess, frame_number,
                                                                               atari.state, 
                                                                               main_dqn,
                                                                               evaluation=True)
                processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                if gif: 
                    frames_for_gif.append(new_frame)
                if terminal:
                    eval_rewards.append(episode_reward_sum)
                    gif = False # Save only the first game of the evaluation as a gif
                     
            print("Evaluation score:\n", np.mean(eval_rewards))       
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], PATH)
            except IndexError:
                print("No evaluation game finished")
            
            #Save the network parameters
            saver.save(sess, PATH+'/'+model_name, global_step=frame_number)
            frames_for_gif = []
            
            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(frame_number, np.mean(eval_rewards), file=eval_reward_file)


# Code from https://gist.github.com/iganichev/d2d8a0b1abc6b15d4a07de83171163d4
# Load variables in if they are relevant
# Currently transfers conv layers
# vars_list is a set of tf.trainable_variables
def optimistic_restore(session, save_file, vars_list):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                      var in vars_list
                      if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                          tf.global_variables()),
                      tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
                
    print("Restoring the following variables: {}".format(restore_vars))
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

# Train on GPU if a trained_path is provided
def transfer_initialized_train_dqn(main_dqn,  main_dqn_vars, param_summaries, target_dqn=None, target_dqn_vars=[], trained_path = None, save_file = None, model_name="my_model"):
    # Trained path: The path (if provided) to look for the saved file from. EG: "trained/pong/"
    # Save_file: The path (if provided) to save tf outputs to
    # The model name
    """Contains the training and evaluation loops"""
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)   # (★)
    if (target_dqn != None):
        network_updater = TargetNetworkUpdater(main_dqn_vars, target_dqn_vars)
    action_getter = ActionGetter(atari.env.action_space.n, 
                                 replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
                                 max_frames=MAX_FRAMES)

    with tf.Session() as sess:
        
        if trained_path != None:
            # Init everything to start off unitialized weights
            sess.run(init)
            # Load saver for weights that can be transferred
            # saver = tf.train.import_meta_graph(trained_path+save_file)
            optimistic_restore(sess, tf.train.latest_checkpoint(trained_path), main_dqn_vars+target_dqn_vars)
        
        else:
            sess.run(init)
        
        frame_number = 0
        rewards = []
        loss_list = []
        
        while frame_number < MAX_FRAMES:
            ########################
            ####### Training #######
            ########################
            epoch_frame = 0
            while epoch_frame < EVAL_FREQUENCY:
                terminal_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                for _ in range(MAX_EPISODE_LENGTH):
                    # (4★)
                    action = action_getter.get_action(sess, frame_number, atari.state, main_dqn)   
                    # (5★)
                    processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)  
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward
                    
                    # (7★) Store transition in the replay memory
                    my_replay_memory.add_experience(action=action, 
                                                    frame=processed_new_frame[:, :, 0],
                                                    reward=reward, 
                                                    terminal=terminal_life_lost)   
                    
                    if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        loss = learn_double_dqn(sess, my_replay_memory, main_dqn, target_dqn,
                                     BS, gamma = DISCOUNT_FACTOR) # (8★)
                        loss_list.append(loss)
                    if target_dqn != None and frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        network_updater.update_networks(sess) # (9★)
                    
                    if terminal:
                        terminal = False
                        break

                rewards.append(episode_reward_sum)
                
                # Output the progress:
                if len(rewards) % 10 == 0:
                    # Scalar summaries for tensorboard
                    if frame_number > REPLAY_MEMORY_START_SIZE:
                        summ = sess.run(param_summaries, 
                                        feed_dict={LOSS_PH:np.mean(loss_list), 
                                                   REWARD_PH:np.mean(rewards[-100:])})
                        
                        SUMM_WRITER.add_summary(summ, frame_number)
                        loss_list = []
                    # Histogramm summaries for tensorboard
                    summ_param = sess.run(param_summaries)
                    SUMM_WRITER.add_summary(summ_param, frame_number)
                    
                    print(len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open('rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number, 
                              np.mean(rewards[-100:]), file=reward_file)
            
            ########################
            ###### Evaluation ######
            ########################
            terminal = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluate_frame_number = 0
            
            for _ in range(EVAL_STEPS):
                if terminal:
                    terminal_life_lost = atari.reset(sess, evaluation=True)
                    episode_reward_sum = 0
                    terminal = False
               
                # Fire (action 1), when a life was lost or the game just started, 
                # so that the agent does not stand around doing nothing. When playing 
                # with other environments, you might want to change this...
                action = 1 if terminal_life_lost else action_getter.get_action(sess, frame_number,
                                                                               atari.state, 
                                                                               main_dqn,
                                                                               evaluation=True)
                processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                if gif: 
                    frames_for_gif.append(new_frame)
                if terminal:
                    eval_rewards.append(episode_reward_sum)
                    gif = False # Save only the first game of the evaluation as a gif
                     
            print("Evaluation score:\n", np.mean(eval_rewards))       
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], PATH)
            except IndexError:
                print("No evaluation game finished")
            
            #Save the network parameters
            saver.save(sess, PATH+'/'+model_name, global_step=frame_number)
            frames_for_gif = []
            
            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(frame_number, np.mean(eval_rewards), file=eval_reward_file)