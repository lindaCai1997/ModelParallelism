import networkx
import itertools  
import operator
import heapq
import math

def topo(G, P, max_size):
    sorted_nodes = list(networkx.topological_sort(G))

    i = 1
    for node in sorted_nodes:
        if i > len(P):
            i = 1
        G.nodes[node]['p'] = i
        i += 1

    '''
    num_proc = len(P)
    num_nodes = G.number_of_nodes()
    nodes_per_proc = int(math.ceil(float(num_nodes)/num_proc))

    lo = 0
    hi = 0
    for num in range(num_proc):
        curr_mem = 0
        hi = min(num_nodes, nodes_per_proc*(num+1))
        for ID in range(lo, hi, 1):
            curr_mem += G.nodes[sorted_nodes[ID]]['memory']
            if (curr_mem > max_size):
                hi = ID
                break;
        for ID in range(lo, hi, 1):
            curr_id = sorted_nodes[ID]
            G.nodes[curr_id]['p'] = num+1
    lo = hi
    for ID in range(num_nodes):
        curr_id = sorted_nodes[ID]
        if (G.in_degree(curr_id) == 0):
            G.nodes[curr_id]['s'] = 0.0
        else:
            G.nodes[curr_id]['s'] = float('inf')
    for PID in range(num_proc):
        P.nodes[PID+1]['t'] = 0.0

    makespan = 0
    task_list = {}
    for ID in range(num_nodes):
        curr_id = sorted_nodes[ID]    
        curr_node = G.nodes[curr_id]
        pred = list(G.predecessors(curr_id))
        start_time = 0
        for pred_id in pred:
            pred_node = G.nodes[pred_id]
            if (curr_node['p'] == pred_node['p']):
                start_time = max(start_time, pred_node['s'] + pred_node['weight'])
            else:
                start_time = max(start_time, pred_node['s'] + pred_node['weight'] 
					+ G[pred_id][curr_id]['weight']) 
        curr_node['s'] = start_time
        task_list[curr_id] = start_time
    task_list = sorted(task_list.items(), key=operator.itemgetter(1))
    #print task_list
    for ID, start_time in task_list:
        curr_node = G.nodes[ID]
        curr_node['s'] = max(start_time, P.nodes[curr_node['p']]['t'])
        P.nodes[curr_node['p']]['t'] = curr_node['s'] + curr_node['weight']
        makespan = max(makespan, P.nodes[curr_node['p']]['t'])
    print("makespan: ", makespan, "miliseconds")
    '''
    '''
    span = 0
    for T in G.nodes():
        t = G.nodes[T]['t'] + G.nodes[T]['weight']
        span = max(span, t)
    print(''.join(['makespan: ', str(span), ' microseconds']))
    '''
    node = {}
    comp = {}
    memo = {}
    for proc in P:
        node[proc] = 0
        comp[proc] = 0
        memo[proc] = 0
    for T in G.nodes():
        node[G.nodes[T]['p']] += 1
        comp[G.nodes[T]['p']] += G.nodes[T]['weight']
        memo[G.nodes[T]['p']] += G.nodes[T]['memory']
    for key in memo:
        print(''.join(['P', str(key), ': ', str(node[key]), ' nodes, ', str(comp[key]), ' microseconds, ', str(memo[key]), ' bytes']))
    return G, 0


