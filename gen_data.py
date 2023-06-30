import sys
import numpy as np
import pickle
import pandas as pd
import networkx as nx
#import matplotlib.pyplot as plt
import random
import time

class alterArcInfo():
    def __init__(self, node_s_j, node_i, node_s_i, node_j):
        self.node_s_j = node_s_j
        self.node_i = node_i
        self.node_s_i = node_s_i
        self.node_j = node_j
        self.isSet = False
        self.cost1 = -1
        self.cost2 = -1

    def __str__(self):
        output = "Alter arc 1: " + str(self.node_s_j) + "-->" + str(self.node_i) + " || arc 2: " + str(self.node_s_i) + "-->" + str(self.node_j)
        return output


class nodeInfo():
    def __init__(self, node_id, train_id, direction, ord, block, arr_time, dep_time):
        self.node_id = node_id
        self.train_id = train_id
        self.direction= direction
        self.ord = ord
        self.block = block
        self.arr_time = arr_time
        self.dep_time = dep_time
        self.name = train_id + "-"+ str(block)
        self.next_fixed_node = -1
        self.prev_fixed_node = -1
        self.pos_x = 0
        self.pos_y = 0

    def __str__(self):
        output = str(self.name)
        return output

def add_entry_delay(nodes, max_delay, event_pro):
    entry_delay = 0
    for i in range(len(nodes)):
        node = nodes[i]
        if node.ord == 1:
            if np.random.randint(0, 100) < event_pro:
                entry_delay = np.random.randint(0, max_delay)  # 관제 구간진입시 지연발생
            #entry_delay = 0
            #print(entry_delay)
        node.arr_time += entry_delay
        node.dep_time += entry_delay
        #print(node.arr_time)
        #print(node.dep_time)
    return nodes

def gen_nodes(train_file):
    with open(train_file, mode="r") as f:
        lines = f.readlines()

    nodes = []
    node_labels = {}
    nodeMap = {}
    train_ids = []
    node = nodeInfo(0, 'start', 0, -1, 0, 0, 0) #시작더미노드
    node_labels[0] = node.name
    nodes.append(node)
    nodeMap[node.name] = node.node_id
    numNode = 1

    for line_num in range(len(lines)):
        #print(line_num, ",", lines[line_num])
        tmp_strs = lines[line_num].split('\n')
        strs = tmp_strs[0].split('\t')
        #strs = lines[line_num].split('\t')
        train_id = strs[0]
        if train_id not in train_ids:
            train_ids.append(train_id)
        direction = int(strs[1])
        ord = int(strs[2])
        block = strs[3]
        arr = strs[4]
        h, m, s = arr.split(':')
        arr_time = int(h) * 3600 + int(m) * 60 + int(s)
        #print(arr_time)
        dep = strs[5]
        h, m, s = dep.split(':')
        dep_time = int(h) * 3600 + int(m) * 60 + int(s)
        #print(dep_time)

        node = nodeInfo(numNode, train_id, direction, ord, block, arr_time, dep_time)
        node_labels[numNode] = node.name
        nodes.append(node)
        nodeMap[node.name] = node.node_id
        numNode = numNode + 1

    print(train_ids)
    node = nodeInfo(numNode, 'end', 0, -1, 0, 0, 0) #종료더미노드
    nodes.append(node)
    nodeMap[node.name] = node.node_id

    return (nodes, nodeMap, train_ids, len(lines))

def gen_nodes1(train_file, train_list):
    with open(train_file, mode="r") as f:
        lines = f.readlines()

    train_list.append('start')
    train_list.append('end')

    nodes = []
    node_labels = {}
    nodeMap = {}
    node = nodeInfo(0, 'start', 0, -1, 0, 0, 0) #시작더미노드
    node_labels[0] = node.name
    nodes.append(node)
    nodeMap[node.name] = node.node_id
    numNode = 1

    for line_num in range(len(lines)):
        #print(line_num, ",", lines[line_num])
        strs = lines[line_num].split('\t')
        train_id = strs[0]
        direction = int(strs[1])
        ord = int(strs[2])
        block = strs[3]
        arr = strs[4]
        h, m, s = arr.split(':')
        arr_time = int(h) * 3600 + int(m) * 60 + int(s)
        #print(arr_time)
        dep = strs[5]
        h, m, s = dep.split(':')
        dep_time = int(h) * 3600 + int(m) * 60 + int(s)
        #print(dep_time)
        if train_id not in train_list : continue
        node = nodeInfo(numNode, train_id, direction, ord, block, arr_time, dep_time)
        node_labels[numNode] = node.name
        nodes.append(node)
        nodeMap[node.name] = node.node_id
        numNode = numNode + 1


    node = nodeInfo(numNode, 'end', 0, -1, 0, 0, 0) #종료더미노드
    nodes.append(node)
    nodeMap[node.name] = node.node_id

    return (nodes, nodeMap)

def gen_p_graph(network_file):
    tc_list = []
    p_graph = nx.DiGraph()
    with open(network_file, mode="r") as f:
        lines = f.readlines()

    for line_num in range(len(lines)):
        # print(line_num, ",", lines[line_num])
        strs = lines[line_num].split('\t')
        from_block = strs[0]
        to_block = strs[1]
        direction = int(strs[2])
        distance = int(strs[3])
        p_graph.add_edge(from_block, to_block)
        p_graph[from_block][to_block]['direction'] = direction
        p_graph[from_block][to_block]['weight'] = distance

        if from_block not in tc_list:
            tc_list.append(from_block)

        if to_block not in tc_list:
            tc_list.append(to_block)

    print(tc_list)
    return [p_graph, tc_list]


def gen_train_nodes(n_trains, nodes):
    train_seq = {}
    for m in range(n_trains):
        train_seq[m] = []
    m = 0
    for i in range(len(nodes)):
        if i == len(nodes)-1 : #마지막 열차노드
            continue
        from_node = nodes[i]
        if from_node.ord == -1: continue
        to_node = nodes[i + 1]
        if (from_node.train_id == to_node.train_id):
            train_seq[m].append(from_node.node_id)
        else :
            train_seq[m].append(from_node.node_id)
            m += 1

    #print(train_seq)
    return train_seq

def gen_ag_graph(nodes): #출발기준
    ag_graph = nx.DiGraph()
    # 고정호 추가
    pos_x = -50
    pos_y = 0
    max_ord = 0
    for i in range(len(nodes)):
        from_node = nodes[i]
        if max_ord < from_node.ord:
            max_ord = from_node.ord
        if i == len(nodes)-1 : #마지막 노드 x축 위치 보정
            pos_x =max_ord * 50
        from_node.pos_x = pos_x
        from_node.pos_y = pos_y
        #print(from_node)
        if i == len(nodes)-1 : #마지막 열차노드
            continue
        to_node = nodes[i+1]
        if(from_node.train_id == to_node.train_id and from_node.ord+1 == to_node.ord):
            pos_x = pos_x + 50
            ag_graph.add_edge(from_node.node_id, to_node.node_id)
            from_node.next_fixed_node = to_node.node_id
            to_node.prev_fixed_node = from_node.node_id
            ag_graph[from_node.node_id][to_node.node_id]['weight'] = -(to_node.dep_time-to_node.arr_time)
            ag_graph[from_node.node_id][to_node.node_id]['type'] = 'Fixed'
        else :
            pos_x = 0
            pos_y = pos_y + 50
            if to_node.ord == 1: #각 열차의 첫번째 노드
                ag_graph.add_edge(0, to_node.node_id)
                ag_graph[0][to_node.node_id]['weight'] = -(1*to_node.arr_time) - (to_node.dep_time - to_node.arr_time)
                ag_graph[0][to_node.node_id]['type'] = 'Dummy'

            if to_node.ord == 1 and from_node.ord != -1: #각 열차의 마지막 노드
                ag_graph.add_edge(from_node.node_id, len(nodes)-1)
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = 0
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'

            if to_node.ord == -1:
                ag_graph.add_edge(from_node.node_id, len(nodes)-1) #각 열차의 마지막 노드
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = 0
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'

    return ag_graph

def gen_ag_graph2(nodes): #도착기준
    ag_graph = nx.DiGraph()
    # 고정호 추가
    pos_x = -50
    pos_y = 0
    max_ord = 0
    for i in range(len(nodes)):
        from_node = nodes[i]
        if max_ord < from_node.ord:
            max_ord = from_node.ord
        if i == len(nodes)-1 : #마지막 노드 x축 위치 보정
            pos_x =max_ord * 50
        from_node.pos_x = pos_x
        from_node.pos_y = pos_y
        #print(from_node)
        if i == len(nodes)-1 : #마지막 열차노드
            continue
        to_node = nodes[i+1]
        if(from_node.train_id == to_node.train_id and from_node.ord+1 == to_node.ord):
            pos_x = pos_x + 50
            ag_graph.add_edge(from_node.node_id, to_node.node_id)
            from_node.next_fixed_node = to_node.node_id
            to_node.prev_fixed_node = from_node.node_id
            ag_graph[from_node.node_id][to_node.node_id]['weight'] = -(from_node.dep_time-from_node.arr_time)
            ag_graph[from_node.node_id][to_node.node_id]['type'] = 'Fixed'
        else :
            pos_x = 0
            pos_y = pos_y + 50
            if to_node.ord == 1 : #각 열차의 첫번째 노드
                ag_graph.add_edge(0, to_node.node_id)
                ag_graph[0][to_node.node_id]['weight'] = -(1*to_node.arr_time)
                ag_graph[0][to_node.node_id]['type'] = 'Dummy'

            if to_node.ord == 1 and from_node.ord != -1: #각 열차의 마지막 노드
                ag_graph.add_edge(from_node.node_id, len(nodes)-1)
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = -(from_node.dep_time-from_node.arr_time)
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'

            if to_node.ord == -1:
                ag_graph.add_edge(from_node.node_id, len(nodes)-1) #각 열차의 마지막 노드
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = 0
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'

    return ag_graph


def gen_alter_arcs(p_graph, nodes):
    alterArcSet = []
    for n in p_graph.nodes():
        alternative = []
        for i in range(len(nodes)):
            node = nodes[i]
            if(node.block == n):
                alternative.append(node.node_id)
        #print(n,",", alternative)
        if(len(alternative)>1):
            for i in range(len(alternative)):
                 for j in alternative[i+1:]:
                     if nodes[alternative[i]].next_fixed_node == nodes[j].next_fixed_node : continue
                     node_i = nodes[alternative[i]]
                     node_j = nodes[j]
                     node_s_i = nodes[node_i.next_fixed_node]
                     node_s_j = nodes[node_j.next_fixed_node]
                     alterArcs = alterArcInfo(node_s_j, node_i, node_s_i, node_j)
                     #print(alterArcs)
                     alterArcSet.append(alterArcs)

    return  alterArcSet

def gen_alter_arcs2(p_graph, nodes):
    alterArcSet = []
    for n in p_graph.nodes():
        alternative = []
        for i in range(len(nodes)):
            node = nodes[i]
            if(node.block == n):
                alternative.append(node.node_id)
        #print(n,",", alternative)
        if(len(alternative)>1):
            for i in range(len(alternative)):
                 for j in alternative[i+1:]:
                     if nodes[alternative[i]].next_fixed_node == nodes[j].next_fixed_node : continue
                     node_s_i = nodes[alternative[i]]
                     node_s_j = nodes[j]
                     node_i = nodes[node_s_i.prev_fixed_node]
                     node_j = nodes[node_s_j.prev_fixed_node]
                     #node_i = nodes[alternative[i]]
                     #node_j = nodes[j]
                     #node_s_i = nodes[node_i.next_fixed_node]
                     #node_s_j = nodes[node_j.next_fixed_node]
                     alterArcs = alterArcInfo(node_s_j, node_i, node_s_i, node_j)
                     #print(alterArcs)
                     alterArcSet.append(alterArcs)

    return  alterArcSet

if __name__ == '__main__':
    start_time = time.time()
    p_graph = nx.DiGraph()
    network_file = "large_network.txt"
    with open(network_file, mode="r") as f:
        lines = f.readlines()

    for line_num in range(len(lines)):
        #print(line_num, ",", lines[line_num])
        strs = lines[line_num].split('\t')
        from_block = strs[0]
        to_block = strs[1]
        direction = int(strs[2])
        distance = int(strs[3])
        p_graph.add_edge(from_block, to_block)
        p_graph[from_block][to_block]['direction'] =  direction
        p_graph[from_block][to_block]['weight'] = distance



    train_file = "large_train.txt"
    with open(train_file, mode="r") as f:
        lines = f.readlines()

    nodes = []
    node_labels = {}
    node = nodeInfo(0, 'start', 0, -1, 0, 0, 0) #시작더미노드
    node_labels[0] = node.name
    nodes.append(node)
    numNode = 1
    for line_num in range(len(lines)):
        #print(line_num, ",", lines[line_num])
        strs = lines[line_num].split('\t')
        train_id = strs[0]
        direction = int(strs[1])
        ord = int(strs[2])
        block = strs[3]
        arr = strs[4]
        h, m, s = arr.split(':')
        arr_time = int(h) * 3600 + int(m) * 60 + int(s)
        #print(arr_time)
        dep = strs[5]
        h, m, s = dep.split(':')
        dep_time = int(h) * 3600 + int(m) * 60 + int(s)
        #print(dep_time)
        node = nodeInfo(numNode, train_id, direction, ord, block, arr_time, dep_time)
        node_labels[numNode] = node.name
        nodes.append(node)
        numNode = numNode + 1


    node = nodeInfo(numNode, 'end', 0, -1, 0, 0, 0) #종료더미노드
    node_labels[numNode] = node.name
    nodes.append(node)

    ag_graph = nx.DiGraph()
    # 고정호 추가
    pos_x = 0
    pos_y = 0
    pos = {}
    for i in range(len(nodes)):
        from_node = nodes[i]
        pos[i] = [pos_x, pos_y]
        from_node.pos_x = pos_x
        from_node.pos_y = pos_y
        #print(from_node)
        if i == len(nodes)-1 : #마지막 열차노드
            continue
        to_node = nodes[i+1]
        if(from_node.train_id == to_node.train_id and from_node.ord+1 == to_node.ord):
            pos_x = pos_x + 50
            ag_graph.add_edge(from_node.node_id, to_node.node_id)
            from_node.next_fixed_node = to_node.node_id
            ag_graph[from_node.node_id][to_node.node_id]['weight'] = -(from_node.dep_time-from_node.arr_time)
            ag_graph[from_node.node_id][to_node.node_id]['type'] = 'Fixed'
        else :
            pos_x = 0
            pos_y = pos_y + 50
            if to_node.ord == 1: #열차의 첫번째 노드
                ag_graph.add_edge(0, to_node.node_id)
                ag_graph[0][to_node.node_id]['weight'] = 0
                ag_graph[0][to_node.node_id]['type'] = 'Dummy'

            if to_node.ord == 1 and from_node.ord != -1:
                ag_graph.add_edge(from_node.node_id, len(nodes)-1)
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = 0
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'

            if to_node.ord == -1:
                ag_graph.add_edge(from_node.node_id, len(nodes)-1)
                ag_graph[from_node.node_id][len(nodes)-1]['weight'] = 0
                ag_graph[from_node.node_id][len(nodes)-1]['type'] = 'Dummy'


    #대안호 정보추가
    alterArcSet = []
    for n in p_graph.nodes():
        alternative = []
        for i in range(len(nodes)):
            node = nodes[i]
            if(node.block == n):
                alternative.append(node.node_id)
        #print(n,",", alternative)
        if(len(alternative)>1):
            for i in range(len(alternative)):
                 for j in alternative[i+1:]:
                     #print(alternative[i],",",j)
                     if nodes[alternative[i]].next_fixed_node == -1 or nodes[j].next_fixed_node == -1 : continue
                     node_i = nodes[alternative[i]]
                     node_j = nodes[j]
                     node_s_i = nodes[node_i.next_fixed_node]
                     node_s_j = nodes[node_j.next_fixed_node]
                     #print(node_s_j.name,"-->",node_i.name)
                     #print(node_s_i.name,"-->",node_j.name)
                     alterArcs = alterArcInfo(node_s_j, node_i, node_s_i, node_j)
                     #print(alterArcs)
                     alterArcSet.append(alterArcs)

                     #ag_graph.add_edge(node_s_j.node_id, node_i.node_id)
                     #ag_graph[node_s_j.node_id][node_i.node_id]['weight'] = -120
                     #ag_graph[node_s_j.node_id][node_i.node_id]['type'] = 'Alter'

                     #ag_graph.add_edge(node_s_i.node_id, node_j.node_id)
                     #ag_graph[node_s_i.node_id][node_j.node_id]['weight'] = -120
                     #ag_graph[node_s_i.node_id][node_j.node_id]['type'] = 'Alter'

    eFixed = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Fixed']
    eAlter = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Alter']
    eDummy = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Dummy']

    if nx.negative_edge_cycle(ag_graph) == False:
        longest_path = nx.bellman_ford_path(ag_graph, source=0, target=21, weight='weight')
        print(longest_path)
        print(nx.path_weight(ag_graph, longest_path, weight = 'weight'))


    #print(list(nx.simple_cycles(ag_graph)))
    #print(list(nx.find_cycle(ag_graph, 0, 'original')))
    #print(nx.negative_edge_cycle(ag_graph))
    #print(nx.shortest_path(ag_graph, source=0, target=21, weight = 'weight'))
    #print(nx.bellman_ford_path(ag_graph, source=0, target=21, weight='weight'))


    nx.draw_networkx_nodes(ag_graph, pos = pos, node_size=200)
    nx.draw_networkx_edges(ag_graph, pos = pos, edgelist=eFixed, width=1)
    nx.draw_networkx_edges(ag_graph, pos = pos, edgelist=eAlter, width=1, alpha=0.5, edge_color="b", style="dashed")
    nx.draw_networkx_edges(ag_graph, pos=pos, edgelist=eDummy, width=1, alpha=0.5, edge_color="r", style="dashed")
    nx.draw_networkx_labels(ag_graph, pos, node_labels)
    plt.axis("off")
    plt.show()

    #A = nx.to_numpy_matrix(ag_graph)
    A = nx.to_numpy_array(ag_graph)
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] < 0:
                A[i][j] = 1

    print(A)

    print(time.time()-start_time)