import numpy as np
# Biblioteca para trabalhar com grafos
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import random as rd
from assignment import assigment



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
            G.edges[i, j]['weight'] = G.edges[j, i]['weight'] = dist_matrix[i][j]
    return G


def NearestNeighbor(mat):
    # assert isinstance(mat, np.matrix), "use np.matrix"
    n = len(mat)
    used = np.zeros(n, dtype=bool)
    used[0] = True
    cost = 0
    last = 0
    route = [0]
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
            route.append(last)
            used[last] = True

    cost += mat[last, 0]
    return cost, np.array(route, dtype=int)


def two_opt(route, mat):
    def opt2CostDelta(mat, i, ni, j, nj):
        return mat[i, j] + mat[ni, nj] - mat[i, ni] - mat[j, nj]

    improved = True
    changed = True #iniciar false
    n = len(route)
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                if opt2CostDelta(mat, route[i], route[i + 1], route[j], route[j + 1]) < 0:
                    route[i + 1:j + 1] = route[j:i:-1]
                    improved = True
        for i in range(1, n - 2):
            if opt2CostDelta(mat, route[i], route[i + 1], route[-1], route[0]) < 0:
                np.copyto(route, np.roll(route, -1))
                route[i: -1] = route[n - 2:i - 1:-1]
                np.copyto(route, np.roll(route, 1))
                improved = True
        if improved:
            changed = True
    return changed #retornar changed e a route


def cost(route, mat):
    c = mat[route[0], route[-1]]
    for i in range(1, len(route)):
        c += mat[route[i - 1], route[i]]
    return c


def heuristica1(mat):
    n = len(mat)
    matb = np.zeros([n, n])
    dcost, x, v, u = assigment(mat)
    pi = np.zeros(n)
    for i in range(n):
        pi[i] = -0.5 * (v[i] + u[i])
    for i in range(n):
        for j in range(i):
            matb[i, j] = matb[j, i] = np.round(mat[i, j] + pi[i] + pi[j], 3)
    ub, route = NearestNeighbor(matb)
    two_opt(route, matb) 
    return cost(route, mat), route 


def heuristica2(mat, pi):
    n = len(mat)
    matb = np.zeros([n, n])
    for i in range(n):
        for j in range(i):
            matb[i, j] = matb[j, i] = np.round(mat[i, j] + pi[i] + pi[j], 3)
    ub, route = NearestNeighbor(matb)
    two_opt(route, mat) #condição para saber se melhorou diante da rota anterior
    return cost(route, mat), route #fazer um return na qual retorna a rota gerada na busca local two_opt e o custo desta


# def heuristica3(mat, pi):  # muito ruim
#     dist = []
#     for i in range(len(mat)):
#         for j in range(i):
#             dist.append([i, j, mat[i, j] + pi[i] + pi[j]])
#             dist.append([j, i, mat[i, j] + pi[i] + pi[j]])
#
#     fitness_dists = mlrose.TravellingSales(distances=dist)
#     problem_fit = mlrose.TSPOpt(length=len(mat), fitness_fn=fitness_dists, maximize=False)
#     best_state, best_fitness = mlrose.genetic_alg(problem_fit, max_attempts=100, random_state=7)
#     route = np.roll(best_state, -np.argmin(best_state))
#     return cost(route, mat), route


def route_to_graph(route, graph):
    graph.clear_edges()
    for i in range(0, len(route)):
        if route[i - 1] < route[i]:
            graph.add_edge(route[i - 1], route[i])
        else:
            graph.add_edge(route[i], route[i - 1])
    return


def graph_to_route(graph):
    n = len(graph.edges)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    route = [0]
    p = 0
    added = True
    while added:
        added = False
        for i in graph.neighbors(p):
            if not visited[i]:
                route.append(i)
                p = i
                visited[i] = True
                added = True
                break

    return route
