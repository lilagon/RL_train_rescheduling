import sys
import numpy as np
import random
import copy

import gen_data as gen
import AG as ag
import networkx as nx


class Env():
    cnt_in_episode = 0
    current_lp_cost = -1
    bigM = 99999

    def __init__(self, network_file, train_file, max_delay):
        self.p_graph, self.tc_list = gen.gen_p_graph(network_file)
        self.n_blocks = nx.number_of_nodes(self.p_graph)

        self.nodes, self.nodeMap, self.train_ids, self.n_operations  = gen.gen_nodes(train_file)

        self.n_max_trains = len(self.train_ids)
        self.n_trains = len(self.train_ids)
        self.max_delay = max_delay

        self.num_nodes = len(self.nodes)
        self.train_nodes = gen.gen_train_nodes(self.n_trains, self.nodes)
        self.current_postion = self.cal_postion(self.train_nodes)
        self.next_pos = self.next_postion(self.train_nodes)
        self.train_file = train_file

        self.ag_graph = gen.gen_ag_graph(self.nodes)
        self.alterArcSet = gen.gen_alter_arcs2(self.p_graph, self.nodes)
        self.alterArcSet = ag.fixed_alter_arcs2(self.ag_graph, self.alterArcSet)
        ag.print_alter_arcs(self.alterArcSet)
        self.node_labels = {}
        self.pos = {}
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            self.pos[i] = [node.pos_x, node.pos_y]
            self.node_labels[i] = node.name

        self.mip = 0
        self.n_Feasible = 0
        self.n_Fail = 0
        self.n_Optimal = 0

    def cal_postion(self, train_nodes):
        #print(train_nodes)
        postion = []
        for i in range(self.n_trains):
            #print(self.nodes[train_nodes[i][0]].block)
            if len(train_nodes[i]) == 0:
                postion.append(0)
            else :
                postion.append(self.nodes[train_nodes[i][0]].block)
        return postion

    def total_postion(self, train_nodes):
        # print(train_nodes)
        total_pos = np.zeros(self.n_blocks)
        for i in range(self.n_trains):
            if len(train_nodes[i]) != 0:
                for j in range(len(train_nodes[i])):
                    tc_index = self.tc_list.index(self.nodes[train_nodes[i][j]].block)
                    total_pos[tc_index] += 1
        #print(total_pos)
        total_pos = total_pos / self.n_trains
        return total_pos

    def next_postion(self, train_nodes):
        #print(train_nodes)
        postion = []
        for i in range(self.n_trains):
            #print(self.nodes[train_nodes[i][0]].block)
            if len(train_nodes[i]) - 1 <= 0:
                postion.append(0)
            else : postion.append(self.nodes[train_nodes[i][1]].block)
        return postion

    def direction_postion(self, train_nodes):
        up_direction = np.zeros(self.n_blocks)
        dn_direction = np.zeros(self.n_blocks)
        for i in range(self.n_trains):
            if len(train_nodes[i]) != 0:
                tc_index = self.tc_list.index(self.nodes[train_nodes[i][0]].block)
                if self.nodes[train_nodes[i][0]].direction == 1:
                    up_direction[tc_index] = 1
                if self.nodes[train_nodes[i][0]].direction == 2:
                    dn_direction[tc_index] = 1
        #print(total_pos)
        return (up_direction, dn_direction)


    def reset(self):
        #self.n_trains = np.random.randint(2, self.n_max_trains+1)
        self.n_trains = self.n_max_trains
        self.train_list = random.sample(self.train_ids, self.n_trains)
        #print(self.n_trains, " : ", self.train_list)
        self.cnt_in_episode = 0
        self.nodes, self.nodeMap = gen.gen_nodes1(self.train_file, self.train_list)
        self.num_nodes = len(self.nodes)
        self.nodes = gen.add_entry_delay(self.nodes, self.max_delay, 100)
        self.train_nodes = gen.gen_train_nodes(self.n_trains, self.nodes)
        self.current_postion = self.cal_postion(self.train_nodes)
        self.next_pos = self.next_postion(self.train_nodes)
        self.ag_graph = gen.gen_ag_graph(self.nodes)
        self.alterArcSet = gen.gen_alter_arcs2(self.p_graph, self.nodes)
        self.alterArcSet = ag.fixed_alter_arcs2(self.ag_graph, self.alterArcSet)

        self.node_labels = {}
        self.pos = {}
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            self.pos[i] = [node.pos_x, node.pos_y]
            self.node_labels[i] = node.name

        self.current_lp_cost = -1*ag.cost(self.ag_graph)
        self.current_state = self.ag_to_state()
        self.mip = 0
        #print(self.current_state)
        #print(self.current_lp_cost)
        # sys.exit()

        return (self.current_state)

    def ag_to_state(self):
        self.current_postion = self.cal_postion(self.train_nodes)
        self.next_pos = self.next_postion(self.train_nodes)

        #print(self.current_postion)
        states0 = nx.to_numpy_array(self.p_graph) #물리적 네트워크
        states1 = np.zeros(self.n_blocks) #Block 점유여부
        for i in self.current_postion:
            if i == 0: continue  # current_postion이 없는경우
            tc_index = self.tc_list.index(i)
            states1[tc_index] = 1

        states3 = np.zeros(self.n_blocks) # Block 가장 빠른 도착시간
        states4 = np.zeros(self.n_blocks)  # Block 가장 빠른 출발시간
        states10 = np.zeros(self.n_blocks)  # AG 해당 노드 in-degree
        states11 = np.zeros(self.n_blocks)  # AG 해당 노드 out-degree

        entry_times = nx.to_numpy_array(self.ag_graph)[0]
        min_time = np.min(-1*np.delete(entry_times, np.where(entry_times == 0)))
        #print("entry_times: ",entry_times)
        #print("min_entry_time: ", min_time)

        node_indexs = []
        for i in range(len(self.current_postion)):
            if self.current_postion[i] == 0 :
                node_index = nx.number_of_nodes(self.ag_graph)-1
            else: node_index = self.train_nodes[i][0]
            node_indexs.append(node_index)

        #print("node_indexs: ",node_indexs)
        tc_indexs= []
        if nx.negative_edge_cycle(self.ag_graph) == False:
            longest_paths = nx.single_source_bellman_ford_path_length(self.ag_graph, source=0)

            for i in range(len(self.current_postion)):
                n = self.nodes[node_indexs[i]]
                if self.current_postion[i] == 0: continue # current_postion이 없는경우
                tc_index = self.tc_list.index(self.current_postion[i])
                if tc_index in tc_indexs: continue
                states10[tc_index] = self.ag_graph.in_degree(n.node_id)
                states11[tc_index] = self.ag_graph.out_degree(n.node_id)
                states3[tc_index] = -1 * longest_paths[n.node_id] - min_time
                if n.next_fixed_node != -1:
                    states4[tc_index] = -1 * longest_paths[n.next_fixed_node] - min_time
                else: states4[tc_index] = -1 * longest_paths[n.node_id] + (n.dep_time-n.arr_time) - min_time
                tc_indexs.append(tc_index)

        else:
            for i in range(len(self.current_postion)):
                if self.current_postion[i] == 0: continue
                tc_index = self.tc_list.index(self.current_postion[i])
                if tc_index in tc_indexs: continue
                states10[tc_index] = 0
                states11[tc_index] = 0
                states3[tc_index] = 0
                states4[tc_index] = 0
                tc_indexs.append(tc_index)

        states3 = states3 / max(1, np.max(states3), np.max(states4))
        states4 = states4 / max(1, np.max(states3), np.max(states4))
        states10 = states10 / max(1, np.max(states10), np.max(states11))
        states11 = states11 / max(1, np.max(states11), np.max(states11))

        states5 = np.zeros(self.n_blocks)  # Next Block
        for i in self.next_postion(self.train_nodes):
            if i == 0 : continue #next postion이 없는경우
            tc_index = self.tc_list.index(i)
            states5[tc_index] = 1

        states6 = np.zeros(self.n_blocks) #분기점 tc 정보
        degree = nx.degree(self.p_graph)
        tc = nx.nodes(self.p_graph)
        for i in tc:
            if degree[i] > 4:
                states6[self.tc_list.index(i)] = 1

        states8, states9 = self.direction_postion(self.train_nodes)

        states = np.append(self.n_blocks, states0)
        states = np.append(states, states1)
        states = np.append(states, states3)
        states = np.append(states, states4)
        states = np.append(states, states5)
        states = np.append(states, states6)
        states = np.append(states, states8)
        states = np.append(states, states9)
        states = np.append(states, states10)
        states = np.append(states, states11)

        if self.current_lp_cost != self.bigM:
            states = np.append(states, self.current_lp_cost/self.bigM)
        else : states = np.append(states, 0)
        return states

    def step(self, action):
        #print(action)
        self.cnt_in_episode += 1
        if self.current_lp_cost == self.bigM:
            self.n_Fail += 1
            print("Fail1!")
            return (self.current_state, -1 * self.bigM, True)

        if action == self.n_blocks:
            print("Finish !!!")
            returned_reward = ag.cost(self.ag_graph)

            if self.current_lp_cost == self.bigM:
                self.n_Fail += 1
                print("Fail2!")
                return (self.current_state, -1 * self.bigM, True)

            if self.current_lp_cost != self.bigM and ag.is_select_all_alter_arcs(self.alterArcSet) == True:
                self.n_Feasible += 1
                print("Feasible!", self.mip, self.current_lp_cost)
                if int(self.mip) == self.current_lp_cost:
                    #returned_reward = self.bigM
                    self.n_Optimal += 1
                    print("Optimal!")
                #ag.results(self.ag_graph, self.nodes)
                #ag.slow(self.ag_graph, self.pos, self.node_labels)

            return (self.current_state, returned_reward, True)


        block = self.tc_list[action] #tc명
        avail_times = []
        avail_trains = []
        train_index = -1

        for i in range(len(self.current_postion)):
            if block == self.current_postion[i]:
                avail_times.append(self.nodes[self.train_nodes[i][0]].arr_time)
                avail_trains.append(i)

        if len(avail_times) > 0:
            train_index = avail_trains[np.argmin(avail_times)]

        if train_index == -1:
            return (self.current_state, -1*self.cnt_in_episode, False)


        node_index = self.train_nodes[train_index][0]
        node_alter_set = ag.node_alter_arcs(self.alterArcSet, node_index)

        for t in range(len(node_alter_set)):
            alterArcInfo = node_alter_set[t]

            if alterArcInfo.node_j.node_id == node_index and alterArcInfo.isSet != True:
                self.ag_graph.add_edge(alterArcInfo.node_s_j.node_id, alterArcInfo.node_i.node_id)
                self.ag_graph[alterArcInfo.node_s_j.node_id][alterArcInfo.node_i.node_id]['weight'] = -120
                self.ag_graph[alterArcInfo.node_s_j.node_id][alterArcInfo.node_i.node_id]['type'] = 'Alter'
                alterArcInfo.isSet = True

            if alterArcInfo.node_i.node_id == node_index and alterArcInfo.isSet != True:
                self.ag_graph.add_edge(alterArcInfo.node_s_i.node_id, alterArcInfo.node_j.node_id)
                self.ag_graph[alterArcInfo.node_s_i.node_id][alterArcInfo.node_j.node_id]['weight'] = -120
                self.ag_graph[alterArcInfo.node_s_i.node_id][alterArcInfo.node_j.node_id]['type'] = 'Alter'
                alterArcInfo.isSet = True

        del self.train_nodes[train_index][0]
        old_lp_cost = self.current_lp_cost
        self.current_lp_cost = -1 * ag.cost(self.ag_graph)

        returned_reward = 0
        #ag.slow(self.ag_graph, self.pos, self.node_labels)
        return (self.ag_to_state(), returned_reward, False)





