import sys
import numpy as np
import pickle
import time
import copy
import gen_data as gen
import AG as ag
import networkx as nx
#import matplotlib.pyplot as plt
import time
from gurobipy import *


BigM = 3600

class MIP():
    def __init__(self, network_file, train_file):
        self.p_graph, self.tc_list = gen.gen_p_graph(network_file)
        self.nodes, self.nodeMap, self.train_ids, self.n_operations = gen.gen_nodes(train_file)
        self.ag_graph = gen.gen_ag_graph(self.nodes)
        self.alterArcSet = gen.gen_alter_arcs2(self.p_graph, self.nodes)
        self.alterArcSet = ag.fixed_alter_arcs2(self.ag_graph, self.alterArcSet)
        self.num_alterArcSet = len(self.alterArcSet)
        self.node_labels = {}
        self.pos = {}
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            self.pos[i] = [node.pos_x, node.pos_y]
            self.node_labels[i] = node.name

def mip_cost(alterArcSet, ag_graph, nodes):
    num_nodes = len(nodes)
    num_alterArcSet = len(alterArcSet)

    m = Model("mip")
    m.setParam("Threads", 1)
    #m.setParam("TimeLimit", 120)
    m.setParam('OutputFlag', 0)
    m.setParam('Heuristics', 0)
    m.setParam(GRB.Param.Presolve, 0)
    m.setParam(GRB.Param.Cuts, 0)
    m.setParam(GRB.Param.VarBranch, 1)

    t = m.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='t')
    z = m.addVars(num_alterArcSet, vtype=GRB.BINARY, name='z')

    m.setObjective(t[num_nodes-1]-t[0], GRB.MINIMIZE)

    eFixed = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Fixed' or d["type"] == 'Dummy']
    sub_graph = ag_graph.edge_subgraph(eFixed)

    #eAlter = [(u, v) for (u, v, d) in ag_graph.edges(data=True) if d["type"] == 'Alter']
    #sub_graph2 = ag_graph.edge_subgraph(eAlter)

    m.addConstrs(t[j] -t[i] >= -1*sub_graph[i][j]['weight'] for i in range(num_nodes) for j in range(num_nodes) if sub_graph.has_edge(i, j) == True)

    for s in range(num_alterArcSet):
        k = alterArcSet[s]
        m.addConstr(t[k.node_j.node_id] - t[k.node_s_i.node_id] >= 120 - BigM*z[s])
        m.addConstr(t[k.node_i.node_id] - t[k.node_s_j.node_id] >= 120 - BigM * (1-z[s]))

    #m.write('mip.lp')
    m.optimize()
    print(m.Runtime)

    for v in m.getVars():
        if v.varName[0] == 't' :
            node_idx = int(v.varName[2:v.varName.find(']')])
            print('%s %g %s' % (v.varName, v.x, ag.transTimeToStr(round(v.x))), nodes[node_idx].name)

    return m.objVal


if __name__=='__main__':
    mip = MIP("mid_network.txt", "mid_train-18.txt")
    mip_cost(mip.alterArcSet, mip.ag_graph, mip.nodes)
    num_nodes = len(mip.nodes)


    m = Model("mip")
    m.setParam("Threads", 1)
    m.setParam('OutputFlag', 0)
    m.setParam('Heuristics', 0)
    m.setParam(GRB.Param.MIPFocus, 3)
    t = m.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='t')
    z = m.addVars(mip.num_alterArcSet, vtype=GRB.BINARY, name='z')

    m.setObjective(t[num_nodes-1]-t[0], GRB.MINIMIZE)

    eFixed = [(u, v) for (u, v, d) in mip.ag_graph.edges(data=True) if d["type"] == 'Fixed' or d["type"] == 'Dummy']
    sub_graph = mip.ag_graph.edge_subgraph(eFixed)

    eAlter = [(u, v) for (u, v, d) in mip.ag_graph.edges(data=True) if d["type"] == 'Alter']
    sub_graph2 = mip.ag_graph.edge_subgraph(eAlter)


    m.addConstrs(t[j] -t[i] >= -1*sub_graph[i][j]['weight'] for i in range(num_nodes) for j in range(num_nodes) if sub_graph.has_edge(i, j) == True)

    for s in range(mip.num_alterArcSet):
         k = mip.alterArcSet[s]
         #print(k.node_j.node_id)
         m.addConstr(t[k.node_j.node_id] - t[k.node_s_i.node_id] >= 120 - BigM*z[s])
         m.addConstr(t[k.node_i.node_id] - t[k.node_s_j.node_id] >= 120 - BigM * (1-z[s]))

    m.write('mip.lp')
    m.optimize()

    print(m.objVal)

    # for i in range(mip.num_nodes):
    #     print(mip.nodes[i],"-->", ag.transTimeToStr(round(t[i].x)))

    #for v in m.getVars():
    #    if v.varName[0] == 't' :
    #        print('%s %g %s' % (v.varName, v.x, ag.transTimeToStr(round(v.x))))
