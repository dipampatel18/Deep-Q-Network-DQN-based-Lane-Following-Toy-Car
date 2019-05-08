#!/usr/bin/env python3

from __future__ import print_function

import rospy
from std_msgs.msg import String
from std_msgs.msg import String
from sensor_msgs.msg import Image
from toy_car_rl.srv import *

import numpy as np
import tensorflow as tf      # Deep Learning library
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames
from collections import deque# Ordered collection with ends
import warnings # This ignore all the warning messages that are normally printed during the tra
warnings.filterwarnings('ignore')
import pickle
import threading

from dqn import *

class rl_sub():
    def __init__(self):
        self.sub = rospy.Subscriber("picam_image", Image, self.callback)
        self.rewards_service = rospy.Service("rwd_srv", rwd, self.reward_change)
        self.pub_command = rospy.Publisher("toy_car_cmd", String, queue_size=10)

        #stacking Initialization
        self.stack_size = 4
        self.stacked_frames  =  deque([np.zeros((480,640), dtype=np.int) for i in range(self.stack_size)], maxlen=4) 
        
        # Environment Params
        self.left = [1, 0, 0]
        self.right = [0, 0, 1]
        self.gas = [0, 1, 0]
        self.possible_actions = [self.left, self.right, self.gas]

        ### MODEL HYPERPARAMETERS
        self.state_size = [480,640,4]
        self.action_size = len(self.possible_actions)
        self.learning_rate =  0.0002 #alpha

        ### TRAINING HYPERPARAMETERS
        # self.total_episodes = 500        # Total episodes for training
        # self.max_steps = 100              # Max possible steps in an episode
        self.batch_size = 64

        # Exploration parameters for epsilon greedy strategy
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability 
        self.decay_rate = 0.0001            # exponential decay rate for exploration prob

        # Q learning hyperparameters
        self.gamma = 0.95
        
        ### MEMORY HYPERPARAMETERS
        self.pretrain_length = self.batch_size   # Number of experiences stored in the Memory when initialized for the first time
        self.memory_size = 1000000          # Number of experiences the Memory can keep 

        tf.reset_default_graph()

        # Instantiate the DQNetwork
        self.network = DQNetwork(self.state_size, self.action_size, self.learning_rate)

        # Instantiate memory
        self.memory = Memory(max_size = self.memory_size)

        #reward state
        self.reward_status = "p"

        #For saving purpose
        self.saver = tf.train.Saver()

        self.state = np.zeros((480, 640), dtype=np.int)

        # Loader Flags
        self.tf_load_flag = False
        self.memory_load_flag = False

        self.writer = tf.summary.FileWriter("./tensorboard/dqn/1")

        self.write_op = tf.summary.merge_all()
        
    def reward_change(self, data):
        self.reward_status = data.rw_type
        return rwdResponse(True)

    def stack_frames(self, state, is_new_episode):
        # Preprocess frame
        frame = self.preprocess_frame(state)
        
        if is_new_episode:
            # Clear our self.stacked_frames
            self.stacked_frames = deque([np.zeros((480,640), dtype=np.int) for i in range(self.stack_size)], maxlen=4)
            
            # Because we're in a new episode, copy the same frame 4x
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            self.stacked_frames.append(frame)
            
            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=2)
            
        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_frames, axis=2) 
        
        return stacked_state

    def preprocess_frame(self, frame):
        normalized_frame = frame/255.0    
        return normalized_frame

    def callback(self, data):
        # rospy.loginfo("CHANGED STATE!")
        state = np.asarray([x for x in data.data], dtype=np.int)
        self.state = state.reshape((480, 640))
        
    
    def main(self):
        rospy.loginfo("MAIN LOOP!")
        episode = 0
        while not rospy.is_shutdown():
            episode += 1
            while self.reward_status != "p":
                pass
            with tf.Session() as sess:
                if self.tf_load_flag:
                    saver.restore(sess, "./tmp/model.ckpt")
                else:
                    sess.run(tf.global_variables_initializer())

                if self.memory_load_flag:
                    with open('filename_pi.obj', 'r') as filehandler:
                        self.memory = pickle.load(filehandler)
                else:
                    pass
                               
                decay_step = 0
                # Initialize the rewards of the episode
                episode_rewards = []
                # Make a new episode and observe the first state
                state = self.state
                # Stacked State
                state = self.stack_frames(state, True)

                done = False
                rospy.loginfo("MAIN LOOP 2!")
                self.pub_command.publish("f")
                while not done:
                    decay_step += 1
                    action, explore_probability = self.predict_action(sess, self.explore_start, self.explore_stop, self.decay_rate, decay_step, state, self.possible_actions)
                    if action == self.left:
                        self.pub_command.publish("l")
                    elif action == self.right:
                        self.pub_command.publish("r")
                    elif action == self.gas:
                        self.pub_command.publish("f")
                    reward = self.get_reward()
                    done = self.check_done()
                    rospy.logdebug("DONE %s" % done)
                    print(done)
                    episode_rewards.append(reward)
                    if done:
                        self.pub_command.publish("s")
                        # the episode ends so no next state
                        next_state = np.zeros((480,640), dtype=np.int)
                        next_state = self.stack_frames(next_state, False)

                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        print('Total reward: {}'.format(total_reward))
                        
                        self.memory.add((state, action, reward, next_state, done))
                        rospy.loginfo("EPISODE FINISHED!")
                    else:
                        next_state = self.state
                        next_state = self.stack_frames(next_state, False)
                        self.memory.add((state, action, reward, next_state, done))
                        state = next_state
                        rospy.loginfo("EPISODE CONTINUED!")
                    
                    if len(self.memory.buffer) > 64:
                        rospy.loginfo("TRAINING")
                        batch = self.memory.sample(self.batch_size)
                        states_mb = np.array([each[0] for each in batch], ndmin=3)
                        actions_mb = np.array([each[1] for each in batch])
                        print("ACTION")
                        print(actions_mb)
                        rewards_mb = np.array([each[2] for each in batch]) 
                        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                        dones_mb = np.array([each[4] for each in batch])
                        rospy.loginfo("AFTER TRAINING")
                        target_Qs_batch = []
                        rospy.loginfo("BEFORE SESS")
                        Qs_next_state = sess.run(self.network.output, feed_dict = {self.network.inputs_: next_states_mb})
                        rospy.loginfo("AFTER SESS")
                        for i in range(0, len(batch)):
                            terminal = dones_mb[i]
                            rospy.loginfo("IN LOOP TRAINING")

                            # If we are in a terminal state, only equals reward
                            if terminal:
                                rospy.loginfo("IF LOOP TRAINING")
                                target_Qs_batch.append(rewards_mb[i])
                                
                            else:
                                rospy.loginfo("ELSE LOOP TRAINING")
                                target = rewards_mb[i] + self.gamma * np.max(Qs_next_state[i])
                                target_Qs_batch.append(target)

                        rospy.loginfo("PST TRAINING")                                

                        targets_mb = np.array([each for each in target_Qs_batch])

                        loss, _ = sess.run([self.network.loss, self.network.optimizer],
                                            feed_dict={self.network.inputs_: states_mb,
                                                    self.network.target_Q: targets_mb,
                                                    self.network.actions_: actions_mb})

                        # Write TF Summaries
                        # summary = sess.run(self.write_op, feed_dict={self.network.inputs_: states_mb,
                        #                                 self.network.target_Q: targets_mb,
                        #                                 self.network.actions_: actions_mb})
                        # self.writer.add_summary(summary, episode)
                        # self.writer.flush()

                        #Writting memory
                        rospy.loginfo("SAVING")
                        with open('filename_pi.obj', 'wb') as file_pi:
                            pickle.dump("memory.obj", file_pi)


                        if episode % 5 == 0:
                            save_path = saver.save(sess, "./models/model.ckpt")
                            print("Model Saved")

    
    def get_reward(self):
        if self.reward_status == "n":
            return -10
        elif self.reward_status == "p":
            return 5
        else:
            return 0
    
    def check_done(self):
        if self.reward_status != "p":
            return True
        else:
            return False

    def predict_action(self, sess, explore_start, explore_stop, decay_rate, decay_step, state, actions):
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            action = random.choice(self.possible_actions)
            
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = self.possible_actions[int(choice)]
                    
        return action, explore_probability

def runner(): 
    while not rospy.is_shutdown():
        rospy.spin()

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('pic_listner', anonymous=True)
    th = threading.Thread(target=runner)
    th.daemon = True
    th.start()
    ic = rl_sub()
    ic.main() 

    # spin() simply keeps python from exiting until this node is stopped
    th.join()

if __name__ == '__main__':
    listener()