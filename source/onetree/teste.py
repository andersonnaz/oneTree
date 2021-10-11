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
    assert isinstance(mat, np.matrix), "use np.matrix"
    n = len(mat)
    used = np.zeros(n, dtype=bool)
    used[0] = True
    cost = 0
    last = 0
    path = [0]
    for i in range(1, n):
        minarg = -1
        min = np.inf
        for j in range(1, n):
            if (not used[j] and mat[last, j] < min):
                min = mat[last, j]
                minarg = j
        if minarg >= 0:
            cost += min
            last = minarg
            path.append(last)
            used[last] = True

    cost += mat[last, 0]
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
        G.nodes[i]["coord"] = coordinates[i]
        G.nodes[i]["label"] = i
        for j in range(i):
            G.edges[i,j]['weight'] = G.edges[j,i]['weight'] = dist_matrix[i][j]
    return G

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
    assert isinstance(mat, np.matrix), "use np.matrix"
    n = Gm1.number_of_nodes()
    for i in range(n):
        for j in range(i):
            Gm1.edges[i, j]['weight'] = Gm1.edges[j, i]['weight'] = mat[i, j] + pi[i] + pi[j]

    # gerando a árvore com o restando dos nós.
    spaningTree = nx.minimum_spanning_tree(Gm1, algorithm="kruskal", weight='weight')
    # nx.draw_networkx(spaningTree, spaningTree.nodes.data('pos'), with_labels=True, node_size=200, font_size=8)
    # localizar as menores arestas que conectam o nó removido na matriz de distâncias
    vet = np.zeros(n)
    for i in range(n):
        vet[i] = mat[n,i] + pi[i] + pi[n]
    a, b = np.argpartition(vet, 2)[:2]

    # inserindo o nó removido
    oneT.clear_edges()
    oneT.add_edges_from(spaningTree.edges)
    oneT.add_edge(n, a)
    oneT.add_edge(n, b)
    return

def w_funcPi(graph, mat, pi):
    assert isinstance(mat, np.matrix), "use np.matrix"
    cost = 0
    for i, j in graph.edges:
        cost += mat[i,j] + pi[i] + pi[j]
    return cost

def w_func(graph, mat):
    assert isinstance(mat, np.matrix), "use np.matrix"
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

def subGrad(ub, G, mat, LAMBDA=1.61, eps=1e-4):
    n = G.number_of_nodes()
    # one tree
    oneT = nx.Graph.copy(G)
    # G menos o último vertice
    Gm1 = nx.Graph.copy(G)
    Gm1.remove_node(n - 1)
    # vetor de pesos
    pi = np.zeros(n)
    # melhor vetor de pesos encontrado
    bestPi = np.zeros(n)

    L = 2
    oneTree(oneT, Gm1, mat, pi)
    maxW = w = w_funcPi(oneT, mat, pi)
    # número de iterações
    ite = 1
    while (L > eps):
        d = degree(oneT)
        if all_equals(d, 2):
            print("Ciclo Hamiltoniano encontrado.")
            break
        # subgradiente
        g = d - 2
        # passo
        # if t == -1:
        t = ((ub + 2 * sum(bestPi)) - w) / (np.linalg.norm(g) ** 2)
        while L > eps:
            ite += 1
            pi = pi + L * t * g
            oneTree(oneT, Gm1, mat, pi)
            w = w_funcPi(oneT, mat, pi)
            if w > maxW + eps:
                np.copyto(bestPi, pi)
                maxW = w
                break
            else:
                L /= LAMBDA
                np.copyto(pi, bestPi)
                w = maxW
        print("%-6d UBgap:%7.2lf%% \t|g|:%.2lf  \tL:%.5lf" % (ite, 100 * w / ub, np.linalg.norm(g), L))

    oneTree(oneT, Gm1, mat, bestPi)
    return bestPi, oneT, w

def compareSolutions(oficialSolution, oneT, w):
    lista = list(oneT.edges)
    arr = list(np.array(oficialSolution.tours[0]) - 1)

    oficialSolutionEdges = [(arr[-1], arr[0])]
    for i in range(len(arr) - 1):
        j = i + 1
        oficialSolutionEdges.append((arr[i], arr[j]))

    print("Optional solution", lista)
    print('______________________________________________________________________________')
    oficialSolutionEdges = [(i,j) if i<j else (j,i) for i,j in oficialSolutionEdges];#list comprehension
    print("Optimal solution sort", oficialSolutionEdges)
    print('______________________________________________________________________________')
    intersectionEdges = set(lista) & set(oficialSolutionEdges);
    print("edges intersection", intersectionEdges);
    print('problem Length:' , len(lista))
    print('intersection Edges Length:' , len(intersectionEdges))
    print('percent EdgesIn Optimum:', ((len(intersectionEdges)*100)/len(oficialSolutionEdges)))
    w_oficial = 0
    print(mat)
    print(oficialSolutionEdges)
    for i, j in oficialSolutionEdges:
        print("aqui")
        w_oficial += mat[i,j]
    print("OficialSolution:", w_oficial)
    print("OneTreeSolution:", w)
    print('Gap of Function:', w_oficial - w)

rd.seed(7)

G = tsplib95.load('/home/naz/Downloads/rollout_tsp/instances/tsp_data/ulysses16.tsp').get_graph(True)
optimalSolution = tsplib95.load('/home/naz/Downloads/rollout_tsp/instances/tsp_opt/ulysses16.opt.tour')


# G = randomGraph(50)


mat = nx.to_numpy_matrix(G)
ub, path = NearestNeighbor(mat)

pi, oneT, w = subGrad(ub, G, mat, LAMBDA=1.1)
compareSolutions(optimalSolution, oneT, w)

nx.draw_networkx(oneT, oneT.nodes.data('coord'), with_labels=True, node_size=100, font_size=8)
plt.show()