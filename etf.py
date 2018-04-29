import itertools  
import operator
import heapq

class CMPNode():
	def __init__(self, node):
		self.node = node

	def __lt__(self, other):
		return self.node['f'] < other.node['f']

def etf(_G, _P):

	G = _G
	P = _P

	#initialize
	I = list(P.nodes())
	A = [T for T in G.nodes if G.in_degree[T] == 0]
	for t in G.nodes:
		G.nodes[t]['d'] = 0
	done = 0
	CM = 0
	NM = float('Inf')
	t_heap = []
	heapq.heapify(t_heap)
	pairs = list(itertools.product(A, I))
	R = {}
	for pair in pairs:
		R[pair] = 0

	#Main loop
	while done < G.number_of_nodes():
		while len(I) != 0 and len(A) != 0:
			R_copy = R.copy()
			R_list = sorted(R_copy.items(), key=operator.itemgetter(1))
			e = max(CM, R_list[0][1])

			if e <= NM:
				t = G.nodes[R_list[0][0][0]]
				p = P.nodes[R_list[0][0][1]]
				t['p'] = p['id']
				t['s'] = e
				t['f'] = t['s'] + t['weight']
				cmp_t = CMPNode(t)
				for i in I:
					if (t['id'], i) in R:
						del R[(t['id'], i)]
				for a in A:
					if (a, p['id']) in R:
						del R[(a, p['id'])]
				A.remove(t['id'])
				I.remove(p['id'])
				heapq.heappush(t_heap, cmp_t)
	                        
				done += 1
				if t['f'] <= NM:
					NM = t['f']
			else:
				break

	        
		finished_t = []
		while(True):
			if len(list(t_heap)) == 0:
				NM = float('Inf')
				break
			min_t = heapq.heappop(t_heap)
	                
			if min_t.node['f'] == NM:
				finished_t.append(min_t)
			else:
				CM = NM
				NM = min_t.node['f']
				heapq.heappush(t_heap, min_t)
				break

		available_t = []
		for t in finished_t:
			I.append(t.node['p'])
			for successor in G.neighbors(t.node['id']):
				s = G.nodes[successor]
				s['d'] += 1
				if s['d'] == G.in_degree[successor]:
					available_t.append(s)

		for t in available_t:
			A.append(t['id'])
	        for t in A:
			#available processors?
			for p in I:
				if G.in_degree[t] == 0:
					R[(t, p)] = 0
				else:
					res = []
					for s in G.in_edges(t):
						s = s[0]
						f = G.nodes[s]['f']
						n = G[s][t]['weight']
						r = P[G.nodes[s]['p']][p]['weight']
						res.append(f + n * r)
					R[(t, p)] = max(res)

	#testing
	for t in G.nodes():
		print(''.join([str(t), ':', str(G.nodes[t]['p'])]))
	print(max([G.nodes[i]['f'] for i in G.nodes()]))

	return G














