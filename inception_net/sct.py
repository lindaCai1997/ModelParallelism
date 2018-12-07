import networkx as nx
import scipy as sc
from scipy.optimize import linprog
from etf import etf 
#from cylp.cy import CyClpSimplex
from cvxopt import spmatrix, matrix, solvers

def to_float(arr):
	return [float(temp) for temp in arr]
def append_val(col, row_count, val, LHS_VAL, LHS_I, LHS_J):
	LHS_J.append(row_count)
	LHS_I.append(col)
	LHS_VAL.append(val)	

def small_example():
	'''
	G = nx.DiGraph()
	G.add_node(1, weight=3, favor=-1, parent= 0, id=1)
	G.add_node(2, weight=5, favor=-1, parent= 0, id=2)
	G.add_node(3, weight=6, favor=-1, parent= 0, id=3)
	G.add_node(4, weight=4, favor=-1, parent= 0, id=4)
	G.add_node(5, weight=4, favor=-1, parent= 0, id=5)
	G.add_node(6, weight=3, favor=-1, parent= 0, id=6)
	G.add_node(7, weight=4, favor=-1, parent= 0, id=7)
	G.add_edge(1, 3, weight=2, id=1)
	G.add_edge(1, 4, weight=3, id=2)
	G.add_edge(2, 4, weight=2, id=3)
	G.add_edge(2, 5, weight=2, id=4)
	G.add_edge(3, 6, weight=3, id=5)
	G.add_edge(3, 7, weight=3, id=6)
	G.add_edge(4, 6, weight=1, id=7)
	G.add_edge(4, 7, weight=3, id=8)
	'''
	P = nx.DiGraph()
	P.add_node(1, id=1, l=-1, s='free', size=0.0)
	P.add_node(2, id=2, l=-1, s='free', size=0.0)
	P.add_node(3, id=3, l=-1, s='free', size=0.0)
	P.add_edge(1, 1, weight=0)
	P.add_edge(2, 2, weight=0)
	P.add_edge(3, 3, weight=0)
	P.add_edge(1, 2, weight=1)
	P.add_edge(1, 3, weight=1)
	P.add_edge(2, 3, weight=1)
	P.add_edge(2, 1, weight=1)
	P.add_edge(3, 1, weight=1)
	P.add_edge(3, 2, weight=1)
	
	G = nx.DiGraph()
	G.add_node(1, weight=6, id=1, memory=1.0, parent=0)
	G.add_node(2, weight=7, id=2, memory=1.0, parent=0)
	G.add_node(3, weight=9, id=3, memory=1.0, parent=0)
	G.add_node(4, weight=8, id=4, memory=1.0, parent=0)
	G.add_node(5, weight=10, id=5, memory=1.0, parent=0)
	G.add_node(6, weight=6, id=6, memory=1.0, parent=0)
	G.add_node(7, weight=6, id=7, memory=1.0, parent=0)
	G.add_node(8, weight=10, id=8, memory=1.0, parent=0)
	G.add_node(9, weight=6, id=9, memory=1.0, parent=0)

	G.add_edge(1, 3, weight=1, id=1)
	G.add_edge(1, 4, weight=5, id=2)
	G.add_edge(1, 5, weight=2, id=3)
	G.add_edge(2, 3, weight=1, id=4)
	G.add_edge(2, 4, weight=1, id=5)
	G.add_edge(2, 5, weight=1, id=6)
	G.add_edge(3, 6, weight=5, id=7)
	G.add_edge(3, 7, weight=3, id=8)
	G.add_edge(4, 6, weight=4, id=9)
	G.add_edge(4, 7, weight=1, id=10)
	G.add_edge(4, 8, weight=5, id=11)
	G.add_edge(5, 9, weight=1, id=12)

	return G, P


def sct(_G, _P, max_size):
	
	G = _G
	P = _P
	'''
	G, P = small_example()
	'''	

	#calculate favorite child
	favor_child = lp(G)
	idx = 0
	for node in G.nodes:
		G.nodes[node]['favor'] = -1
	for (i, j) in G.edges:
		if favor_child[idx] == 0:
			G.nodes[i]['favor'] = j
		idx += 1
	
	#sets
	S = []
	R = [T for T in G.nodes if G.in_degree[T] == 0]
	ready = {}
	urgent = {}
	
	#main loop
	time = 0
	while(len(S) != G.number_of_nodes()):
		#calculate ready time and urgent time
		for T in R:
			for proc in P.nodes:
				r_time = 0
				for pre in G.in_edges(T):
					pre = pre[0]
					t = G.nodes[pre]['t']
					p = G.nodes[pre]['weight']
					c = G[pre][T]['weight']
					if G.nodes[pre]['p'] == proc:
						r_time = max(r_time, t + p)
					else:
						r_time = max(r_time, t + p + c)
				ready[(T, proc)] = r_time
		
		for T in R:
			u_time = 0
			for pre in G.in_edges(T):
				pre = pre[0]
				t = G.nodes[pre]['t']
				p = G.nodes[pre]['weight']
				c = G[pre][T]['weight']
				u_time = max(u_time, t + p + c)
			urgent[T] = u_time
	
		#calculate state of processor
		for proc in P.nodes:
			p = P.nodes[proc]
			if p['l'] != -1:
				T = p['l']
				if G.nodes[T]['t'] + G.nodes[T]['weight'] > time:
					p['s'] = 'busy'
				elif G.nodes[T]['favor'] != -1:
					favor = G.nodes[T]['favor']
					if favor in R and ready[(favor, proc)] < urgent[favor]:
						p['s'] = 'awake'	
					else:
						p['s'] = 'free'
				else:
						p['s'] = 'free'		
		#	print(p)
		#schedule
		avail = []
		for proc in P.nodes:
			p = P.nodes[proc]
			#for free processor
			if p['s'] == 'free':
				for T in R:
					if p['size'] + G.nodes[T]['memory'] > max_size:
						continue
					if ready[(T, proc)] <= time:
						G.nodes[T]['t'] = time
						G.nodes[T]['p'] = proc
						S.append(T)
						R.remove(T)
						p['l'] = T
						p['size'] += G.nodes[T]['memory']
						avail.append(T)
						break

			#for awake processor
			if p['s'] == 'awake':	
				for T in R:
					if p['size'] + G.nodes[T]['memory'] > max_size:
						continue
					if urgent[T] <= time or (G.nodes[p['l']]['favor'] == T and ready[(T, proc)] <= time):
						G.nodes[T]['t'] = time
						G.nodes[T]['p'] = proc
						S.append(T)
						R.remove(T)
						p['l'] = T
						p['size'] += G.nodes[T]['memory']
						avail.append(T)
						break
			
		#advance time
		new_t = float('Inf')
		for T in S:
			new_t_p = G.nodes[T]['t'] + G.nodes[T]['weight']
			# if T == 23:
				# print(new_t_p)
			if new_t_p > time and new_t_p < new_t:
				new_t = new_t_p
		for T in R:
			for proc in P:
				new_t_p = ready[(T, proc)]
				if new_t_p > time and new_t_p < new_t:
					new_t = new_t_p
		for T in R:
			new_t_p = urgent[T]
			if new_t_p > time and new_t_p < new_t:
				new_t = new_t_p
		time = new_t
		
		#add to R
		for T in avail:
			for suc in G.neighbors(T):
				s = G.nodes[suc]
				s['parent'] += 1
				if s['parent'] == G.in_degree[suc]:
					R.append(suc)
	
	#report
	#for t in G.nodes():
		#print(''.join([G.nodes[t]['name'], ':', str(G.nodes[t]['p'])]))
	#makespan
	span = 0
	for T in G.nodes():
		t = G.nodes[T]['t'] + G.nodes[T]['weight']
		span = max(span, t)
	print(''.join(['makespan: ', str(span), ' microseconds']))
	#computation time and memory on each processors
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
	
	return G, span
	
	
def lp(G):
	'''
		[e1,e2,e3,...,n1,n2,n3,...,w]
	'''
	#preprocess
	num_node = G.number_of_nodes()
	num_edge = G.number_of_edges()

	LHS_VAL = []
	LHS_I = []
	LHS_J = []
	RHS = []	
	#rule 1
	row_count = 0
	for i in range(num_edge):
		append_val(i, row_count, 1, LHS_VAL, LHS_I, LHS_J)
		RHS.append(1)
		row_count += 1
		
	for i in range(num_edge):
		#eq = [0]*(num_node + num_edge + 1)
		#eq[i] = -1
		#LHS.append(eq)
		append_val(i, row_count, -1, LHS_VAL, LHS_I, LHS_J)
		RHS.append(0)
		row_count += 1
	
	#bound_e = [(0,1)] * num_edge
	
	#rule 2
	for i in range(num_node):
		#eq = [0]*(num_node + num_edge + 1)
		#eq[num_edge + i] = -1
		#LHS.append(eq)
		append_val(num_edge + i, row_count, -1, LHS_VAL, LHS_I, LHS_J)
		RHS.append(0)
		row_count += 1

#	bound_n = [(0, None)] * num_node
#	bound = bound_e + bound_n + [(None, None)]
	
	#rule 3

	for (i, j) in G.edges:
		#eq = [0] * (num_node + num_edge + 1)
		#eq[num_edge + i - 1] = 1
		#eq[num_edge + j - 1] = -1
		#eq[G[i][j]['id'] - 1] = G[i][j]['weight']
		#LHS.append(eq)
		append_val(num_edge + i - 1, row_count, 1, LHS_VAL, LHS_I, LHS_J)
		append_val(num_edge + j - 1, row_count, -1, LHS_VAL, LHS_I, LHS_J)
		append_val(G[i][j]['id'] -1, row_count, G[i][j]['weight'], LHS_VAL, LHS_I, LHS_J)
		RHS.append(-G.node[i]['weight'])
		row_count += 1
	#rule 4
	for n in G.nodes:
		if G.out_degree[n] > 0:
			#eq = [0] * (num_node + num_edge + 1)
			for (i, j) in G.out_edges(n):	
				append_val(G[i][j]['id'] - 1, row_count, -1, LHS_VAL, LHS_I, LHS_J)	
				#eq[G[i][j]['id'] - 1] = -1
			#LHS.append(eq)
			RHS.append(1 -  G.out_degree[n])
			row_count += 1
	
	#rule 5
	for n in G.nodes:
		if G.in_degree[n] > 0:
			#eq = [0] * (num_node + num_edge + 1)
			for (i, j) in G.in_edges(n):
				append_val(G[i][j]['id'] - 1, row_count, -1, LHS_VAL, LHS_I, LHS_J)
				#eq[G[i][j]['id'] - 1] = -1
			#LHS.append(eq)
			RHS.append(1 -  G.in_degree[n])  
			row_count += 1
	
	#rule 6
	for n in G.nodes:
		#eq = [0] * (num_node + num_edge + 1)
		#eq[num_edge + n - 1] = 1
		#eq[-1] = -1
		append_val(num_edge + n - 1, row_count, 1, LHS_VAL, LHS_I, LHS_J)
		append_val(num_node + num_edge, row_count, -1, LHS_VAL, LHS_I, LHS_J)
		#LHS.append(eq)
		RHS.append(-G.node[n]['weight'])
		row_count += 1
	
	#solve LP
	goal = [0] * (num_node + num_edge + 1)
	goal[-1] = 1
	#for i in range(len(LHS)):
	#	print(LHS[i], RHS[i])
	# s = CyClpSimplex()
	#print "hello solver!"
	#print "LHSS:", LHS
	goal = matrix(to_float(goal))
	#goal = matrix(float(goal))
	LHS = spmatrix(LHS_VAL, LHS_J, LHS_I)
	RHS = matrix(to_float(RHS))
	#print "goal", goal
	#print "LHS", LHS.size
	#print "RHS", RHS
	#print "bounds", bound
	print("hello solver!")
	solvers.options['maxiters'] = 10000
	solvers.options['refinement'] = 1
	res = solvers.lp(goal, LHS, RHS, solver='mosek')

	print("goodbye solver!")
        # res = sc.optimize.linprog(goal, A_ub=LHS, b_ub=RHS, bounds=bound, method='interior-point')
	#print(res)
	#print res
	#print res['x']
	x = res['x']
	for i in range(len(x)):
		if x[i] < 0.5:
			x[i] = 0
		else:
			x[i] = 1
	
	#print(x)
	return x[:num_edge]
	
#lp(G)
#G = []
#P = []
#sct(G, P, 1e8)
#for g in G.nodes:
#	print(G.nodes[g])


