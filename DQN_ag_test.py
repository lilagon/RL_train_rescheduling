import sys
import logging
import os
import multiprocessing as mp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import copy
import random
from collections import deque
import numpy as np
import AG as ag
import mip as m
import gen_data as gen
import time

from environment_ag_20220402 import Env
from keras.layers import Dense
#from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from keras.models import Sequential
from keras import regularizers
from keras import backend as K

from prioritized_memory import Memory

# K.set_image_dim_ordering('th')


class Agent:

    def __init__(self, n_blocks, n_trains, n_node, n_alterArcSet, load_file=None):

        self.n_blocks = n_blocks
        self.n_trains = n_trains
        self.n_node = n_node

        self.n_alterArcSet = n_alterArcSet

        self.state_size = 2 + self.n_blocks * self.n_blocks + 9 * self.n_blocks

        self.action_size = self.n_blocks + 1

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.model = self.build_model ()
        self.states, self.actions, self.rewards = [], [], []

        self.epsilon = 0.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.decay_freq = 16
        self.discount_factor = 0.9
        self.batch_size = 256
        self.train_start = 20000
        self.train_freq = 16
        self.update_freq = 64
        self.learning_rate = 0.001

        self.memory_size = 20000000
        self.memory = Memory(self.memory_size)

        self.model = self.build_model()
        if load_file is not None:
            print("Load weights: ")
            self.model.load_weights(load_file)

        self.target_model = self.build_model()
        self.update_target_model()
        self.states, self.actions, self.rewards = [], [], []

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = Sequential()
        model.add(Dense(1024,
                        input_dim=self.state_size,
                        activation='relu',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),
                        kernel_initializer="random_uniform",
                        bias_initializer="zeros"))
        model.add(Dense(512,
                        activation='relu',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.add(Dense(256,
                        activation='relu',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.add(Dense(128,
                        activation='relu',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.add(Dense(self.action_size,
                        activation='linear',kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01),
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #model.summary()
        return model


    def append_sample(self, state, action, reward, next_state, is_terminal):
        state_ = copy.deepcopy(state)
        normalized_state = np.reshape(self.state_normalization(state_),[1,agent.state_size])
        old_val = self.model.predict(normalized_state)[0][action]

        if is_terminal:
           pass
        else:
           next_state_ = copy.deepcopy(next_state)
           normalized_next_state = np.reshape(self.state_normalization(next_state_),[1,agent.state_size])
           target_val = self.target_model.predict(normalized_next_state)[0]

        if not is_terminal:
           new_val = reward + self.discount_factor*np.max(target_val)
        else:
           new_val = reward

        error = np.abs(old_val - new_val)
        #print(error)
        self.memory.add(error, (state, action, reward, next_state))

    def get_action(self, state_, current_postion_, next_postion_):
        state = copy.deepcopy (state_)
        current_postion = copy.deepcopy (current_postion_)
        next_postion = copy.deepcopy(next_postion_)
        #print(current_postion)
        remove_set = {0}
        #postion_without_zero = [i for i in self.current_postion if i not in remove_set]
        postion_without_zero = list(set(current_postion) - set(remove_set))
        if np.random.rand () <= self.epsilon:
            self.action_type="exploration"
            avail_positions = []
            for i in range(len(current_postion)):
                tc = current_postion[i]
                if tc != 0 and next_postion[i] not in postion_without_zero:
                    tc_index = env.tc_list.index(tc)
                    avail_positions.append(tc_index)
            if len(avail_positions) == 0: return self.n_blocks #종료
            randomIndex = random.randrange(0, len(avail_positions))
            return avail_positions[randomIndex]

        else:
            self.action_type="exploitation"
            normalized_state = self.state_normalization (state)
            normalized_state = np.reshape (normalized_state, [1, agent.state_size])
            q_values = self.model.predict (normalized_state, verbose=0)[0]
            avail_values = []
            avail_positions = []
            for i in range(len(current_postion)):
                tc = current_postion[i]
                if tc != 0 and next_postion[i] not in postion_without_zero:
                    tc_index = env.tc_list.index(tc)
                    avail_values.append(q_values[tc_index])
                    avail_positions.append(tc_index)

            if len(avail_positions) == 0: return self.n_blocks #종료
            max_position = np.argmax (avail_values)
            return avail_positions[max_position]

    def state_normalization(self, state_):
        state = copy.deepcopy(state_)
        #max_state = np.max(state)
        #state = state / max_state
        return state.reshape((1, self.state_size))

    def train_model(self, global_step, done):
        if self.epsilon >= self.epsilon_min and global_step % self.decay_freq  == 0:
            self.epsilon *= self.epsilon_decay

        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        # mini_batch = np.array(mini_batch).transpose()
        # print(mini_batch[0])
        # sys.exit()

        # print("mini_batch", mini_batch)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards = [], []

        for i in range(self.batch_size):
            #print("mini_batch[i][0]", mini_batch[i][0])
            states[i] = self.state_normalization(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = self.state_normalization(mini_batch[i][3])


        target = self.model.predict(states)
        target_value = self.target_model.predict(next_states)
        errors = []
        # print(target.shape)
        # print(target_value.shape)

        for i in range(self.batch_size):
            absolute_td_error = np.abs(target[i][actions[i]]-rewards[i]+self.discount_factor*(np.amax(target_value[i])))
            if done:
               errors.append(np.abs(target[i][actions[i]]-rewards[i]))
               target[i][actions[i]] = rewards[i]
            else:
               errors.append(np.abs(target[i][actions[i]]-rewards[i]-self.discount_factor*np.amax(target_value[i])))
               target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_value[i]))

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx,errors[i])

        hist = self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        print("===== MSE loss:",hist.history['loss'])


if __name__ == "__main__":

    print("Number of processors: ", mp.cpu_count())

    load_file = None
    if len (sys.argv) == 6:
        print("python DQN_ag_test.py [network_file] [train_file] [max_delay]")
        load_file = sys.argv[5]

    #network_file = "seoul_network.txt"
    #train_file = "seoul_train-10.txt"
    #max_delay = 300

    network_file = sys.argv[1]
    train_file = sys.argv[2]
    max_delay = int(sys.argv[3])


    load_file = "./save_model/20220402_DQN_49_12_9600.h5"

    #for max_delay in max_delays:
    #    for train_file in train_files:
    env = Env(network_file, train_file, max_delay)

    n_blocks = len(env.tc_list)
    n_max_trains = len(env.train_ids)
    n_node = len(env.nodes)
    n_alterArcSet = len(env.alterArcSet)

    agent = Agent (n_blocks, n_max_trains, n_node, n_alterArcSet, load_file)
    f = open("RL_result-" + str(n_blocks) + "-" + str(n_max_trains) + "-" + str(max_delay) + ".txt", 'w')

    scores, episodes, objs = [], [], []
    global_step = 0
    n_optimal = 0

    EPISODES = 1
    ITERATION_PER_EPISODE = n_max_trains*n_blocks

    e = 0
    for e in range(EPISODES, EPISODES + 100):

        e += 1
        #print ("> Episode {0}".format (e))
        state = env.reset ()

        initial_train_nodes = copy.deepcopy(env.train_nodes)
        initial_ag_graph = copy.deepcopy(env.ag_graph)
        initial_alterArcSet = copy.deepcopy(env.alterArcSet)

        initial_obj = env.current_lp_cost
        mip_start_time = time.time()
        env.mip = m.mip_cost(env.alterArcSet, env.ag_graph, env.nodes)
        mip_time = time.time() - mip_start_time

        env.train_nodes = initial_train_nodes
        env.ag_graph = initial_ag_graph
        env.alterArcSet = initial_alterArcSet

        done = False
        current_best = initial_obj
        prev_action = -1
        RL_time = 0
        print("Start-RL")
        for iteration in range(ITERATION_PER_EPISODE):
            global_step += 1
            tmp = time.time()
            action = agent.get_action(state, env.current_postion, env.next_pos)
            RL_time += time.time() - tmp
            next_state, reward, is_terminal = env.step (action)

            pre_action = action

            if is_terminal:
                done = True

            if env.current_lp_cost < current_best: current_best = env.current_lp_cost

            state = copy.deepcopy (next_state)

            if done: break

            # if global_step == 5: sys.exit()
            #print("episode:", e, "global step:", global_step, "iteration", iteration, "action:", action, "reward:",
            #      reward, "current obj.:", env.current_lp_cost,
            #      "action_type:", agent.action_type, "improvement:", initial_obj - env.current_lp_cost, "epsilon:",
            #      agent.epsilon, "best:", current_best, "postion:", env.current_postion)

        objs.append (env.current_lp_cost)
        #print("the end of episode:", e, "\t current obj.:\t", env.current_lp_cost, "\t mip cost:\t", env.mip)
        f.write(str(e) + "\t" + str(env.n_trains) + "\t"+ str(env.current_lp_cost) + "\t" + str(0) + "\t" + str(env.mip) + "\t"+ str(RL_time) + "\t" + str(0) + "\t" + str(mip_time) + "\t" + str(ag.results(env.ag_graph, env.nodes)/env.n_trains) + "\n")
        if env.current_lp_cost == env.mip:
            n_optimal += 1


    print("# Optimal: ", n_optimal)