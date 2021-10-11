# Biblioteca para trabalhar com grafos
import networkx as nx

# Biblioteca para plotar os grafos
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import numpy as np
import random as rd
import tsplib95

def NearestNeighbor(mat):
    n = len(mat[0])
    used = np.zeros(n, dtype=bool)
    used[0] = True
    cost = 0
    last = 0
    path = [0]
    for i in range(1, n):
        minarg = -1
        min = np.inf
        for j in range(1, n):
            if (not used[j] and mat[last][j] < min):
                min = mat[last][j]
                minarg = j
        if minarg >= 0:
            cost += min
            last = minarg
            path.append(last)
            used[last] = True

    cost += mat[last][0]
    return cost, path

def get_distance_matrix(problem):
    n = problem.dimension
    plus = 0
    if next(problem.get_nodes()) == 1:
        plus = 1

    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + plus, n):
            distance_matrix[i, j] = problem.get_weight(i + plus, j + plus)
    
    return distance_matrix

def randomGraph(n, W=1000, H=1000):
    coordinates = []
    for i in range(n):
        coordinates.append((rd.randint(0, W), rd.randint(0, H)))

    dist_matrix = squareform(pdist(coordinates))
    G = nx.complete_graph(n)
    for i in range(n):
        G.nodes[i]["pos"] = coordinates[i]
        G.nodes[i]["label"] = i
    return G, dist_matrix

def testGraph(problem):
    coordinates = list(problem.node_coords.values())
    dist_matrix = squareform(pdist(coordinates))
    #dist_matrix = get_distance_matrix(problem)
    G = nx.complete_graph(problem.dimension)
    
    for i in range(problem.dimension):
        G.nodes[i]["pos"] = coordinates[i]
        G.nodes[i]["label"] = i
    return G, dist_matrix

def oneTree(oneT, Gm1, mat, pi):
    n = Gm1.number_of_nodes()
    for i in range(n):
        for j in range(i):
            Gm1.edges[j, i]['w'] = Gm1.edges[i, j]['w'] = mat[i][j] + pi[i] + pi[j]

    # gerando a árvore com o restando dos nós.
    spaningTree = nx.minimum_spanning_tree(Gm1, algorithm="kruskal", weight='w')
    # nx.draw_networkx(spaningTree, spaningTree.nodes.data('pos'), with_labels=True, node_size=200, font_size=8)
    # localizar as menores arestas que conectam o nó removido na matriz de distâncias
    vet = np.zeros(n)
    for i in range(n):
        vet[i] = mat[n][i] + pi[i] + pi[n]
    a, b = np.argpartition(vet, 2)[:2]

    # inserindo o nó removido
    oneT.clear_edges()
    oneT.add_edges_from(spaningTree.edges)
    oneT.add_edge(n, a)
    oneT.add_edge(n, b)
    return

def w_funcPi(graph, mat, pi):
    cost = 0
    for i, j in graph.edges:
        cost += mat[i][j] + pi[i] + pi[j]
    return cost

def w_func(graph, mat):
    cost = 0
    for i, j in graph.edges:
        cost += mat[i][j]
    return cost

def degree(graph):
    n = graph.number_of_nodes()
    d = np.zeros(n)
    deg = graph.degree()
    for i in range(n):
        d[i] = deg[i]
    return d

def all_equals(vet, x):
    for i in vet:
        if x != i:
            return False
    return True

def oneTree2(oneT, G, mat, pi):
    n = G.number_of_nodes()
    for i in range(n):
        for j in range(i):
            G.edges[j, i]['w'] = G.edges[i, j]['w'] = mat[i][j] + pi[i] + pi[j]

    # gerando a árvore com o restando dos nós.
    spaningTree = nx.minimum_spanning_tree(G, algorithm="kruskal", weight='w')
    # nx.draw_networkx(spaningTree, spaningTree.nodes.data('pos'), with_labels=True, node_size=200, font_size=8)
    oneT.clear_edges()
    oneT.add_edges_from(spaningTree.edges)
    min = np.inf
    argmin = -1
    for i in range(n):
        for j in range(i):
            if not (oneT.has_edge(i, j) or oneT.has_edge(j, i)):
                if min > G.edges[i, j]['w']:
                    min = G.edges[i, j]['w']
                    argmin = (i, j)

    oneT.add_edge(*argmin)
    return

#problem = tsplib95.load(f'instances/tsp_data/burma14.tsp')
#problem = tsplib95.load(f'instances/tsp_data/bayg29.tsp')
#problem = tsplib95.load(f'instances/tsp_data/ulysses22.tsp')
problem = tsplib95.load(f'instances/tsp_data/eil51.tsp')
rd.seed(7)
#G, mat = randomGraph(14)
G, mat = testGraph(problem)

oneT = nx.Graph.copy(G)

Gm1 = nx.Graph.copy(G)
# removendo o nó do Grafo G
Gm1.remove_node(G.number_of_nodes() - 1)

pi = np.zeros(G.number_of_nodes())

# pi = np.random.randint(-5,5,G.number_of_nodes())
# pi[1] = 450
# pi[3] = 100
# pi[0] = 50


##
ub, path = NearestNeighbor(mat)

# x = 0
# for i in range(1, len(path)):
#     x += mat[path[i - 1]][path[i]]
# x += mat[0][path[-1]]


l = 2
bestPi = np.array(pi)
oneTree(oneT, Gm1, mat, pi)
w = w_funcPi(oneT, mat, pi)
maxW = w
minG = np.inf
t = -1
while (True):
    d = degree(oneT)
    if all_equals(d, 2):
        break

    # subgradiente
    g = d - 2
    # passo
    if t == -1:
        tmax = l * ((ub+2*sum(pi)) - w) / (np.linalg.norm(g) ** 2)
        t = tmax
    while t > 1e-6:
        pi = pi + t * g
        # oneTree2(oneT, G, mat, pi)
        oneTree(oneT, Gm1, mat, pi)
        w = w_funcPi(oneT, mat, pi)
        if w > maxW + .001 :
            minG = np.linalg.norm(g)
            bestPi = np.array(pi)
            maxW = w
            break
        else:
            t /= 1.61
            pi = np.array(bestPi)
            w = maxW
    if t <= 1e-6:
        break

    print(w, np.linalg.norm(g), t)


nx.draw_networkx(oneT, oneT.nodes.data('pos'), with_labels=True, node_size=200, font_size=8)
plt.show()
