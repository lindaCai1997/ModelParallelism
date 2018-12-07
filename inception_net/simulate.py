import tensorflow as tf
import networkx as nx
import json
from google.protobuf import text_format
import numpy as np
import re
from etf import *
from sct import *
from topo import *
#from make_plot import plot
import copy
import sys
import time

def get_metadata(graph):
    '''
        Note that many of the nodes are unused, so some of the nodes in the graph 
        may not have data in the dictionaries. Also some nodes don't have shape.
        Another Weird thing is that the three dictionaries are a little different.
    '''

    #import graphi
    '''
    graph_sim = tf.Graph().as_default()
    graph_def = graph_sim.as_graph_def(add_shapes=True)
    with open('./model/cnn_dist_before.pb', 'rb') as f:
        text_format.Merge(f.read(), graph_def)
        tf.import_graph_def(graph_def, name='')
    graph = tf.get_default_graph()
    ''' 

    #get computation time
    comp_dict = {}
    with open ('timeline.json', 'r') as f:
        raw_data = json.load(f)
        data = raw_data['traceEvents']
        size = len(data)
        for i in range(size):
            if 'dur' in data[i]:
                name = data[i]['args']['name']
                duration = data[i]['dur']
                comp_dict[name] = float(duration)

    with open("computation.txt", "w") as f:
        for name in comp_dict:
            f.write(name + ": " + str(comp_dict[name]) + "\n")

    #get memory and shape
    memo_dict = {}
    shape_dict = {}
    with open ('timeline_memory.json', 'r') as f:
        raw_data = json.load(f)
        data = raw_data['traceEvents']
        size = len(data)
        for i in range(size):
            if 'args' in data[i]:
                args = data[i]['args']
                name = data[i]['name']
                if 'snapshot' in args:
                    tensor = args['snapshot']['tensor_description']
                    #get shape
                    tensor_size = re.findall('size: ([0-9]+)', tensor)
                    _prod = 1
                    for s in tensor_size:
                        _prod *= float(s)
                    shape_dict[name] = _prod * 4.0
                    #get memory
                    memo = re.findall('requested_bytes: ([0-9]+)', tensor)
                    if len(memo) == 0:
                        memo_dict[name] = 0.0
                    else:
                        memo_dict[name] = float(memo[0])

    ret = [graph, comp_dict, memo_dict, shape_dict]
    return ret

def create_simulate_graph(graph, comp_dict, memo_dict, shape_dict, num_proc, BANDWIDTH):
    #start to create simulation graph
    G = nx.DiGraph()

    node_dict = {}
    i = 1
    #add node
    sum_weight = 0.0
    for node in graph.get_operations():
        memo = 0
        if node.name in memo_dict:
            memo = memo_dict[node.name]
            sum_weight += memo_dict[node.name]
        if node.name in comp_dict:
            G.add_node(i, weight=comp_dict[node.name], name=node.name, id=i, memory=memo, favor=-1, parent=0)

        else:
            G.add_node(i, weight=1.0, name=node.name, id=i, memory=memo, favor=-1, parent=0)
        node_dict[node.name] = i
        i += 1

    #add edge
    i = 1
    for node in graph.get_operations():
        for _input in node.inputs: 
            input_name = _input.name
            if ":" in input_name:
                index = input_name.index(":")
                input_name = input_name[:index]
            if input_name in shape_dict:
                weight = shape_dict[input_name] * 1e6 / BANDWIDTH  
                G.add_edge(node_dict[input_name], node_dict[node.name], weight=weight, id=i)
            else:
                G.add_edge(node_dict[input_name], node_dict[node.name], weight=0.0, id=i)
            i += 1
    
    return G,sum_weight,node_dict
    

def create_processor_graph(num_proc):
    P = nx.Graph()
    for ID in range(num_proc):
        P.add_node(ID+1, id=ID+1, l=-1, s='free', size=0.0)
    for ID1 in range(num_proc):
        for ID2 in range(num_proc):
            if (ID1 == ID2):
                P.add_edge(ID1+1, ID2+1, weight=0)
            else:
                P.add_edge(ID1+1, ID2+1, weight=1)
    return P

def partition(BANDWIDTH, num_proc, graph, comp_dict, memo_dict, shape_dict, fix_total, arg, method):
    #set the processor config
    G,sum_weight,node_dict = create_simulate_graph(graph, comp_dict, memo_dict, shape_dict, num_proc, BANDWIDTH)
    P = create_processor_graph(num_proc)
    
    #run the algorithm
    print("\n##### Condition: BANDWIDTH ", BANDWIDTH, "  NUM OF PROCS ", num_proc)
    if (fix_total == True):
        if (method == "topo"):
            print("m-TOPO: ")
            _G, span = topo(G, P, float(sum_weight*arg)/num_proc)
        elif (method == "sct"):
            print("m-SCT: ")
            _G, span = sct(G, P, float(sum_weight*arg)/num_proc)
        elif (method == "etf"):
            print("m-ETF: ")
            _G, span = etf(G, P, float(sum_weight*arg)/num_proc)
    else:
        if (method == "topo"):
            print("m-TOPO: ")
            _G, span = topo(G, P, arg)
        elif (method == "sct"):
            print("m-SCT: ")
            _G, span = sct(G, P, arg)
        elif (method == "etf"):
            print("m-ETF: ")
            _G, span = etf(G, P, arg)

    return _G,span,node_dict

def simulate(graph, method, fix_total, arg):
        bandwidth = 4e7
        num_proc = arg
        metadata = get_metadata(graph)
        _G,time,node_dict = partition(bandwidth, num_proc, metadata[0], metadata[1], metadata[2], metadata[3], fix_total, arg, method)
       
        with open('graph_before.txt', 'w') as f:
            for node in _G.nodes():
                n = _G.nodes[node]
                f.write(''.join([n['name'], ',', str(n['p'] - 1), ',', str(n['weight']), ',', str(n['memory']), '\n']))

 
        group_dict = {}
        for node in graph.get_operations():
            group = str(node.colocation_groups()[0])
            index = group.index("@")
            group = group[index+1:-1]
            group_dict[node.name] = group

        f = open('graph_after.txt', 'w')
        for node in graph.get_operations():
            index = node_dict[group_dict[node.name]]
            pro = str(int(_G.nodes[index]['p']) - 1)
            node._set_device(''.join(['/job:worker/task:', pro]))
            

            n = _G.nodes[node_dict[node.name]]
            f.write(''.join([n['name'], ',', pro, ',', str(n['weight']), ',', str(n['memory']), '\n']))            
        f.close()

        for node in graph.get_operations():
            index = node_dict[node.name]
            _G.nodes[index]['p'] = int(node.device[-1]) + 1       

        node = {}
        comp = {}
        memo = {}
        for proc in range(1, arg + 1):
            node[proc] = 0
            comp[proc] = 0
            memo[proc] = 0
        for T in _G.nodes():
            node[_G.nodes[T]['p']] += 1
            comp[_G.nodes[T]['p']] += _G.nodes[T]['weight']
            memo[_G.nodes[T]['p']] += _G.nodes[T]['memory']
        for key in memo:
            print(''.join(['P', str(key), ': ', str(node[key]), ' nodes, ', str(comp[key]), ' microseconds, ', str(memo[key]), ' bytes']))




def experiments(fix_total, arg):
    bandwidths = [1e5, 1e7, 1e9, 1e11]
    num_procs = [3, 6, 9, 12, 15]
    metadata = get_metadata()

    # perform experiments on different number of processors and bandwidth, record the results
    time = np.zeros([3, len(bandwidths), len(num_procs)])

    #methods = ["topo", "etf", "sct"]
    methods = ["etf"]
    for k in range(len(methods)):
        for i in range(len(bandwidths)):
            bandwidth = bandwidths[i]
            for j in range(len(num_procs)):
                num_proc = num_procs[j]
                _G,time[k][i][j] = partition(bandwidth, num_proc, metadata[0], metadata[1], metadata[2], metadata[3], fix_total, arg, methods[k])
                
    if (fix_total):
        np.save("results/topo_time_fix_" + str(arg) + "_times_total", time[0])
        np.save("results/etf_time_fix_" + str(arg) + "_times_total", time[1])
        np.save("results/sct_time_fix_" + str(arg) + "_times_total", time[2])
    else:
        np.save("results/topo_time_fix_each_"+str(arg), time[0])
        np.save("results/etf_time_fix_each_"+str(arg), time[1])
        np.save("results/sct_time_fix_each_"+str(arg), time[2])
    #plot(fix_total,arg)

if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("Usage: ")
        print("If want fixed total memory to be n times graph required memory, python simulate fix n")
        print("If want fixed memory to be M on each machine, python simulate vary M")
        sys.exit() 
    elif (sys.argv[1] == "fix"):
        experiments(True, float(sys.argv[2]))       
    else:
        experiments(False, float(sys.argv[2]))


