import sys
import numpy as np
import pickle
import copy
import networkx as nx
import AG as ag
import matplotlib.pyplot as plt
BigM = -99999

def cost(ag_graph):
    out = 0
    if nx.negative_edge_cycle(ag_graph) == False:
        longest_path = nx.bellman_ford_path(ag_graph, source=0, target=nx.number_of_nodes(ag_graph)-1, weight='weight')
        #print(longest_path)
        out = nx.path_weight(ag_graph, longest_path, weight='weight')
    else : out = BigM
    return out

def add_alter_arc(_ag_graph, node_i, node_j):
    ag_graph = copy.deepcopy(_ag_graph)
    ag_graph.add_edge(node_i.node_id, node_j.node_id)
    ag_graph[node_i.node_id][node_j.node_id]['weight'] = -120
    ag_graph[node_i.node_id][node_j.node_id]['type'] = 'Alter'
    return ag_graph

def is_select_all_alter_arcs(alterArcSet):
    is_select = True
    for s in alterArcSet:
        if s.isSet == False:
            is_select = False
    return is_select

def num_select_alter_arcs(alterArcSet):
    count = 0
    for s in alterArcSet:
        if s.isSet == True:
            count += 1
    return count

def print_alter_arcs(alterArcSet):
    for s in alterArcSet:
        print(s)

def node_alter_arcs(alterArcSet, node):
    node_alter_arcs = []
    for s in alterArcSet:
        if s.node_i.node_id == node or s.node_j.node_id == node :
            #print("find!!",s)
            node_alter_arcs.append(s)
    return node_alter_arcs

def fixed_alter_arcs(ag_graph, alterArcSet): #출발기준
    cnt = 0
    fixed_alter_arc_cost = 0
    for s in alterArcSet :
        #print("original ", s)
        if s.node_s_j.train_id == "end" :
            ag_graph.add_edge(s.node_s_i.node_id, s.node_j.node_id)
            ag_graph[s.node_s_i.node_id][s.node_j.node_id]['weight'] = fixed_alter_arc_cost
            ag_graph[s.node_s_i.node_id][s.node_j.node_id]['type'] = 'Fixed'
            s.isSet = True
            for k in alterArcSet :
                if k.node_i.node_id == s.node_s_i.node_id and k.node_s_j.node_id == s.node_j.node_id :
                    ag_graph.add_edge(k.node_s_i.node_id, k.node_j.node_id)
                    ag_graph[k.node_s_i.node_id][k.node_j.node_id]['weight'] = fixed_alter_arc_cost
                    ag_graph[k.node_s_i.node_id][k.node_j.node_id]['type'] = 'Fixed'
                    k.isSet = True

                if k.node_j.node_id == s.node_s_i.node_id and k.node_s_i.node_id == s.node_j.node_id :
                    ag_graph.add_edge(k.node_s_j.node_id, k.node_i.node_id)
                    ag_graph[k.node_s_j.node_id][k.node_i.node_id]['weight'] = fixed_alter_arc_cost
                    ag_graph[k.node_s_j.node_id][k.node_i.node_id]['type'] = 'Fixed'
                    k.isSet = True
            cnt = cnt + 1

        if s.node_s_i.train_id == "end" :
            ag_graph.add_edge(s.node_s_j.node_id, s.node_i.node_id)
            ag_graph[s.node_s_j.node_id][s.node_i.node_id]['weight'] = fixed_alter_arc_cost
            ag_graph[s.node_s_j.node_id][s.node_i.node_id]['type'] = 'Fixed'
            s.isSet = True
            for k in alterArcSet :
                if k.node_i.node_id == s.node_s_j.node_id and k.node_s_j.node_id == s.node_i.node_id :
                    ag_graph.add_edge(k.node_s_i.node_id, k.node_j.node_id)
                    ag_graph[k.node_s_i.node_id][k.node_j.node_id]['weight'] = fixed_alter_arc_cost
                    ag_graph[k.node_s_i.node_id][k.node_j.node_id]['type'] = 'Fixed'
                    k.isSet = True

                if k.node_j.node_id == s.node_s_j.node_id and k.node_s_i.node_id == s.node_i.node_id :
                    ag_graph.add_edge(k.node_s_j.node_id, k.node_i.node_id)
                    ag_graph[k.node_s_j.node_id][k.node_i.node_id]['weight'] = fixed_alter_arc_cost
                    ag_graph[k.node_s_j.node_id][k.node_i.node_id]['type'] = 'Fixed'
                    k.isSet = True
            cnt = cnt + 1

    alterArcSet = [item for item in alterArcSet if item.isSet != True]

    #for s in alterArcSet:
    #    print("final ", s)

    return alterArcSet

def fixed_alter_arcs2(ag_graph, alterArcSet): #도착기준
    cnt = 0
    fixed_alter_arc_cost = 0
    for s in alterArcSet :
        #print("original ", s)
        #if s.node_s_j.train_id == "end" :
        if s.node_i.train_id == "end":
            ag_graph.add_edge(s.node_s_i.node_id, s.node_j.node_id)
            ag_graph[s.node_s_i.node_id][s.node_j.node_id]['weight'] = fixed_alter_arc_cost
            ag_graph[s.node_s_i.node_id][s.node_j.node_id]['type'] = 'Fixed'
            s.isSet = True
            for k in alterArcSet :
                if k.node_i.node_id == s.node_s_i.node_id and k.node_s_j.node_id == s.node_j.node_id :
                    ag_graph.add_edge(k.node_s_i.node_id, k.node_j.node_id)
                    ag_graph[k.node_s_i.node_id][k.node_j.node_id]['weight'] = fixed_alter_arc_cost
                    ag_graph[k.node_s_i.node_id][k.node_j.node_id]['type'] = 'Fixed'
                    k.isSet = True

                if k.node_j.node_id == s.node_s_i.node_id and k.node_s_i.node_id == s.node_j.node_id :
                    ag_graph.add_edge(k.node_s_j.node_id, k.node_i.node_id)
                    ag_graph[k.node_s_j.node_id][k.node_i.node_id]['weight'] = fixed_alter_arc_cost
                    ag_graph[k.node_s_j.node_id][k.node_i.node_id]['type'] = 'Fixed'
                    k.isSet = True
            cnt = cnt + 1

        if s.node_j.train_id == "end" :
            ag_graph.add_edge(s.node_s_j.node_id, s.node_i.node_id)
            ag_graph[s.node_s_j.node_id][s.node_i.node_id]['weight'] = fixed_alter_arc_cost
            ag_graph[s.node_s_j.node_id][s.node_i.node_id]['type'] = 'Fixed'
            s.isSet = True
            for k in alterArcSet :
                if k.node_i.node_id == s.node_s_j.node_id and k.node_s_j.node_id == s.node_i.node_id :
                    ag_graph.add_edge(k.node_s_i.node_id, k.node_j.node_id)
                    ag_graph[k.node_s_i.node_id][k.node_j.node_id]['weight'] = fixed_alter_arc_cost
                    ag_graph[k.node_s_i.node_id][k.node_j.node_id]['type'] = 'Fixed'
                    k.isSet = True

                if k.node_j.node_id == s.node_s_j.node_id and k.node_s_i.node_id == s.node_i.node_id :
                    ag_graph.add_edge(k.node_s_j.node_id, k.node_i.node_id)
                    ag_graph[k.node_s_j.node_id][k.node_i.node_id]['weight'] = fixed_alter_arc_cost
                    ag_graph[k.node_s_j.node_id][k.node_i.node_id]['type'] = 'Fixed'
                    k.isSet = True
            cnt = cnt + 1

    alterArcSet = [item for item in alterArcSet if item.isSet != True]

    #for s in alterArcSet:
    #    print("final ", s)

    return alterArcSet

def selected_alter_arcs(ag_graph, nodes):
    eAlter = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Alter']
    for e in eAlter :
        print(nodes[e[0]],"-->",nodes[e[1]])

def results(ag_graph, nodes):
    max_delay = 0
    prev_delay = 0
    prev_train = ''
    train_delay = []
    if nx.negative_edge_cycle(ag_graph) == False:
        length, path = nx.single_source_bellman_ford(ag_graph, 0, weight='weight')
        for n in nodes :
            if n.node_id == len(nodes) - 1:
                n.arr_time = 0
                n.dep_time = 0
            re_dep_time = -1*length[n.node_id]
            if n.prev_fixed_node != -1:
                re_arr_time = -1*length[n.prev_fixed_node]
            else : re_arr_time = -1*length[n.node_id] - (n.dep_time-n.arr_time)
            print("(",n.node_id, ")", n, "-->(", transTimeToStr(n.arr_time),"-->", transTimeToStr(n.dep_time),")-->(", transTimeToStr(re_arr_time),"-->", transTimeToStr(re_dep_time),") : ", re_dep_time - n.dep_time)

            strs = n.name.split('-')
            train_name = strs[0]
            delay = re_dep_time - n.dep_time

            if prev_train != train_name :
                train_delay.append(prev_delay)

            prev_train = train_name
            prev_delay = delay

        print("Train delay: ", train_delay)
        print("Max delay: ", max(train_delay))
    return sum(train_delay)

def results2(ag_graph, nodes):
    max_delay = 0
    prev_delay = 0
    prev_train = ''
    train_delay = []
    length, path = nx.single_source_bellman_ford(ag_graph, 0, weight='weight')
    for n in nodes :
        re_arr_time = -1*length[n.node_id]
        if n.next_fixed_node != -1:
            re_dep_time = -1*length[n.next_fixed_node]
        else : re_dep_time = -1*length[n.node_id] + (n.dep_time-n.arr_time)
        print("(",n.node_id, ")", n, "-->(", transTimeToStr(n.arr_time),"-->", transTimeToStr(n.dep_time),")-->(", transTimeToStr(re_arr_time),"-->", transTimeToStr(re_dep_time),")")
        strs = n.name.split('-')
        train_name = strs[0]
        delay = re_dep_time - n.dep_time

        if prev_train != train_name:
            train_delay.append(prev_delay)

        prev_train = train_name
        prev_delay = delay

    print("Train delay: ", train_delay)
    print("Max delay: ", max(train_delay))

def transTimeToStr(time):
    h = int(time / 3600)
    m = int((time - h * 3600) / 60)
    s = int(time - 3600 * h - 60 * m)
    str_time = str(h) + ":" + str(m) +":"+ str(s)
    return str_time


def transStrToTime(str):
    h, m, s = str.split(':')
    time = int(h) * 3600 + int(m) * 60 + int(s)
    return time

def slow(ag_graph, pos, node_labels):
    eFixed = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Fixed']
    eAlter = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Alter']
    eDummy = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Dummy']
    eRelease = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Release']

    nx.draw_networkx_nodes(ag_graph, pos = pos, node_size=250, node_color="white", edgecolors = "orange")
    nx.draw_networkx_edges(ag_graph, pos = pos, edgelist=eFixed, width=1)
    nx.draw_networkx_edges(ag_graph, pos = pos, edgelist=eAlter, width=1, alpha=0.5, edge_color="b", style="dashed")
    nx.draw_networkx_edges(ag_graph, pos=pos, edgelist=eDummy, width=1, alpha=0.5, edge_color="r", style="dashed")
    nx.draw_networkx_edges(ag_graph, pos=pos, edgelist=eRelease, width=1, alpha=0.5, edge_color="r")
    #nx.draw_networkx_labels(ag_graph, pos)
    nx.draw_networkx_labels(ag_graph, pos, node_labels, font_size = 7)
    plt.axis("off")
    plt.show()

def max_cost_alter_arcs(ag_graph, alterArcInfo):
    #print(alterArcInfo)
    ag1 = add_alter_arc(ag_graph, alterArcInfo.node_s_j, alterArcInfo.node_i)
    alterArcInfo.cost1 = ag.cost(ag1)
    #print(alterArcInfo.cost1)
    ag2 = add_alter_arc(ag_graph, alterArcInfo.node_s_i, alterArcInfo.node_j)
    alterArcInfo.cost2 = ag.cost(ag2)
    #print(alterArcInfo.cost2)
    if alterArcInfo.cost1 > alterArcInfo.cost2: return alterArcInfo.cost2
    else : return alterArcInfo.cost1

def remaining_alterArcSet(alterArcSet):
    cnt = 0
    for s in alterArcSet:
        if s.isSet == False:
           cnt+=1
    return cnt/len(alterArcSet)