# Biblioteca para trabalhar com grafos
import networkx as nx

from scipy.sparse.csgraph import minimum_spanning_tree as mst

# Biblioteca para plotar os grafos
import matplotlib.pyplot as plt

import tsplib95 as tsplib

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial import Delaunay
from itertools import combinations
from scipy.optimize import linear_sum_assignment as ap

from assignment import assigment

import numpy as np
import pandas as pd
import random as rd
import dataFrame as df
import os

import tsp

INSTANCES_PATH = '../../instances'

class OneTree():
    def __init__(self, graph, LAMBDA=1.1, minL=1e-3, iniL=2, tol=1e-2, trace=False):
        assert isinstance(graph, nx.classes.graph.Graph), "tipo incorreto para graph"
        self.LAMBDA = LAMBDA
        self.minL = minL
        self.iniL = iniL
        self.tol = tol
        self.trace = trace
        self.__G = graph
        self.original_mat = nx.to_numpy_matrix(self.__G)
        self.__N = len(self.original_mat)
        for i in range(self.__N):
            self.original_mat[i, i] = np.inf
        self.__branch_mat = np.matrix(self.original_mat)
        self.__pi = np.zeros(self.__N)
        self.best_pi = np.zeros(self.__N)
        self.one_tree = nx.Graph.copy(self.__G)
        self.__Gm1 = nx.Graph.copy(self.__G)
        self.__Gm1.remove_node(self.__N - 1)
        return

    def run_one_tree(self, mat, pi):
        one_tree = self.one_tree
        gm1 = self.__Gm1
        n = self.__N - 1
        for i in range(n):
            for j in range(i):
                gm1.edges[i, j]['weight'] = gm1.edges[j, i]['weight'] = mat[i, j] + pi[i] + pi[j]
        # gerando a árvore com o restando dos nós.
        spaningTree = nx.minimum_spanning_tree(self.__Gm1, algorithm="kruskal", weight='weight')
        # nx.draw_networkx(spaningTree, spaningTree.nodes.data('pos'), with_labels=True, node_size=200, font_size=8)
        # localizar as menores arestas que conectam o nó removido na matriz de distâncias
        vet = np.zeros(n)
        for i in range(n):
            vet[i] = mat[n, i] + pi[i] + pi[n]
        a, b = np.argpartition(vet, 2)[:2]
        # inserindo o nó removido
        one_tree.clear_edges()
        one_tree.add_edges_from(spaningTree.edges)
        one_tree.add_edge(n, a)
        one_tree.add_edge(n, b)
        return

    def __cost_one_tree(self, d, mat):
        cost = 0
        for e in self.one_tree.edges:
            cost += mat[e]
        return cost

    def __initial_pi_values(self, mat, pi):
        dcost, x, v, u = assigment(mat)
        for i in range(self.__N):
            pi[i] = -0.5 * (v[i] + u[i])

    def subgradient(self, ub, mat):
        is_ham = False
        l = self.iniL
        self.run_one_tree(mat, self.best_pi)
        d = np.array(self.one_tree.degree())[:, 1]
        max_w = w = self.__cost_one_tree(d, mat) + np.dot(d - 2, self.best_pi)
        if self.trace:
            print("ini w ", w)
        ite = 1
        while (l > self.minL and ub > max_w + self.tol):
            if np.all(d == 2):
                if self.trace:
                    print("Ciclo Hamiltoniano encontrado.")
                is_ham = True
                break
            if ub + 1e-6 < max_w:
                if self.trace:
                    print("UB atingido.")
                break
            # subgradiente
            g = d - 2
            # passo
            t = (ub - w) / (np.linalg.norm(g) ** 2)
            while l > self.minL:
                ite += 1
                self.__pi = self.best_pi + l * t * g
                self.run_one_tree(mat, self.__pi)
                d = np.array(self.one_tree.degree())[:, 1]
                w = self.__cost_one_tree(d, mat) + np.dot(d - 2, self.__pi)
                if w > max_w + self.tol:
                    np.copyto(self.best_pi, self.__pi)
                    max_w = w
                    if self.trace:
                        print("%-6d UBgap:%7.2lf%% \t|g|:%.2lf  \tL:%.9lf" % (
                            ite, 100 * max_w / ub, np.linalg.norm(g), l))
                    break
                else:
                    l /= self.LAMBDA
            # end_while
        # end_while
        self.run_one_tree(mat, self.best_pi)
        return max_w, is_ham

    def __branch(self, X, Y, level=0):
        ec = []
        np.copyto(self.__branch_mat, self.original_mat)
        fixedCost = 0
        for i, j in X:
            self.__branch_mat[i, j] = self.__branch_mat[j, i] = 0
            fixedCost += self.original_mat[i, j]
        for i, j in Y:
            self.__branch_mat[i, j] = self.__branch_mat[j, i] = np.inf

            # [(1, 2), (2, 4), (5, 1)]
            # 1ª => i = 1, j = 2
            # 2ª => i = 2, j = 4
            # 3ª => i = 5, j = 1
            # for i in range(5) => (0, 1, 2, 3, 4)
            # [[[1,4], [2, 4]], [[1,5], [3, 4]]]

        lb, isHam = self.subgradient(self.ub - fixedCost, self.__branch_mat)

        if self.trace:
            print(lb + fixedCost, level, X, '-', Y)

        if lb + fixedCost + self.tol >= self.ub:
            return
        elif isHam:  # novo incumbent
            self.ub = lb + fixedCost
            print('novo incumbent ', self.ub)

            route = tsp.graph_to_route(self.one_tree)
            if tsp.two_opt(route, self.original_mat):
                self.ub = tsp.cost(route, self.original_mat)
                print('local search', self.ub)
                tsp.route_to_graph(route, self.one_tree)
            self.plot_one_tree()

            return
        # gap
        if (self.ub - lb + fixedCost) / self.ub > .01:
            x, route = tsp.heuristica2(self.__branch_mat, self.best_pi)
            if x + fixedCost + self.tol < self.ub:
                self.ub = x + fixedCost
                print("heuristic solution ", self.ub)
                tsp.route_to_graph(route, self.one_tree)
                self.plot_one_tree()

        for e in self.one_tree.edges:
            if e in X or e in Y:
                continue
            i, j = e
            self.__branch_mat[i, j] = self.__branch_mat[j, i] = np.inf
            self.run_one_tree(self.__branch_mat, self.best_pi)
            d = np.array(self.one_tree.degree())[:, 1]
            w = self.__cost_one_tree(d, self.__branch_mat) + np.dot(d - 2, self.best_pi)
            ec.append((w + fixedCost, e))
            self.__branch_mat[i, j] = self.__branch_mat[j, i] = self.original_mat[i, j]
        ec = sorted(ec, reverse=True)
        for i in range(len(ec)):
            if i > 0:
                X.append(ec[i - 1][1])
            Y.append(ec[i][1])
            self.__branch(list(X), Y, level + 1)
            Y.remove(ec[i][1])
        # ec = [[15, (1, 2)], [12, (2, 4)], [10, (6, 8)]]
        # 1ª
        # X = []
        # Y = [(1, 2)]

        # 2ª
        # X = []
        # Y = [(1, 2),(2, 4)]

        # 3ª
        # X = [(1, 2), (2, 4)]
        # Y = [(6, 8)]
        return

    def plot_one_tree(self):
        plt.clf()
        nx.draw_networkx(self.one_tree, self.one_tree.nodes.data('coord'), with_labels=True, node_size=100,
                         font_size=8)
        plt.show()
        # plt.draw()
        # plt.pause(10)

    def branchNBound(self, UB):
        self.ub = UB
        self.__branch([], [])

    def oneTreeModified(self, mat, pi, k):
        edges = set(self.one_tree.edges)
        alternativas = set()
        for edge in edges:
            matb = np.copy(mat)
            pivot = edge
            for j in range(k):
                matb[pivot] = matb[pivot[::-1]] = np.inf
                self.run_one_tree(matb, pi)
                novos = set(self.one_tree.edges) - edges
                for n in novos:
                    alternativas.add(n)
                pivot = list(novos)[0]

        # G.add_edges_from(edges.union(alternativas))
        # nx.draw(G, with_labels=True, node_size=500, node_color='lightgreen')
        # plt.show()
        return edges.union(alternativas)

    def delaunayOneTree(self, graph, mat, pi):
        edges = delaunayTessellation(graph)
        matb = np.copy(mat)
        n = len(mat)

        for i in range(n):
            for j in range(n):
                if not ((i, j) in edges or (j, i) in edges):
                    matb[i, j] = np.inf

        self.run_one_tree(matb, pi)
        return set(self.one_tree.edges)


def delaunayNearestNeighbor(graph, mat):
    edges = delaunayTessellation(graph)

    G = nx.Graph()
    G.add_edges_from(edges)
    nx.draw(G, with_labels=True, node_size=500, node_color='lightgreen')
    plt.show()

    matb = np.copy(mat)
    n = len(matb)
    conjunto = set()

    for i in range(n):
        for j in range(n):
            if not ((i, j) in edges or (j, i) in edges):
                matb[i, j] = np.inf

    used = np.zeros(n, dtype=bool)
    used[0] = True
    last = 0
    route = [0]
    for i in range(n):
        minarg = -1
        min = np.inf
        for j in range(n):
            if (not used[j] and matb[last, j] < min):
                min = matb[last, j]
                minarg = j
        if minarg >= 0:
            last = minarg
            route.append(last)
            used[last] = True

    G.clear_edges()
    for i in range(0, len(route)):
        if route[i - 1] < route[i]:
            G.add_edge(route[i - 1], route[i])
        else:
            G.add_edge(route[i], route[i - 1])

    nx.draw(G, with_labels=True, node_size=500, node_color='lightgreen')
    plt.show()

    return set(G.edges)

    # print(len(d.simplices))
    # print(len(c[d.simplices]))
    # print(c[d.simplices[1,1]])

    # plt.triplot(c[:,0], c[:,1], d.simplices)
    # plt.plot(c[:,0], c[:,1], 'o')
    # plt.show()
    return conjunto


def delaunayTessellation(graph):
    coords = graph.nodes.data('coord')
    c = np.array([coord for i, coord in coords])
    d = Delaunay(c)
    edges = set()
    for triangulo in d.simplices:
        for aresta in combinations(triangulo, 2):
            edges.add(tuple(sorted(aresta)))

    # print(len(d.simplices))
    # print(len(c[d.simplices]))
    # print(c[d.simplices[1,1]])

    # plt.triplot(c[:,0], c[:,1], d.simplices)
    # plt.plot(c[:,0], c[:,1], 'o')
    # plt.show()
    return edges


def nearestNeighbor(graph, mat, k):
    matb = np.copy(mat)
    # assert isinstance(mat, np.matrix), "use np.matrix"
    n = len(matb)
    last = 0
    v = np.zeros(n)
    G = nx.Graph()
    route = [0]

    for x in range(k):
        used = np.zeros(n, dtype=bool)
        used[0] = True
        route.clear()

        for i in range(n):
            minarg = -1
            min = np.inf
            for j in range(n):
                if (not used[j] and matb[last, j] < min):
                    min = matb[last, j]
                    minarg = j
            if minarg >= 0:
                last = minarg
                route.append(last)
                used[last] = True

        for i in range(0, len(route)):
            if route[i - 1] < route[i]:
                G.add_edge(route[i - 1], route[i])
                matb[route[i - 1], route[i]] = np.inf
            else:
                G.add_edge(route[i], route[i - 1])
                matb[route[i], route[i - 1]] = np.inf

    nx.draw(G, with_labels=True, node_size=500, node_color='lightgreen')
    plt.show()

    return set(G.edges)

    # matb = np.copy(mat)
    # n = len(matb)
    # v = np.zeros(n)
    # conjunto = set()
    # flagNode = [False] * n
    # pilha = [0]

    # while(not(np.all(flagNode))):
    #     i = pilha.pop()
    #     for j in range(n):
    #         v[j] = matb[i, j]
    #     for e in np.argpartition(v, k)[:k]:
    #         matb[i,e] = matb[e,i] = np.inf
    #         pilha.append(e)
    #         if not((i, e) in conjunto or (e, i) in conjunto):
    #             conjunto.add((i,e))
    #             flagNode[i] = flagNode[e] = True

    # for i in range(n):
    #     nearestNeighbor_mat[i,i] = np.inf
    #     for j in range(n):
    #         v[j] = nearestNeighbor_mat[i, j]
    #     lista.append(np.argpartition(v, k)[:k])
    # for i in range(n):
    #     for j in range(k):
    #         if not((i, lista[i][j]) in conjunto or (i, lista[i][j])[::-1] in conjunto):
    #             conjunto.add((i, lista[i][j]))


if __name__ == "__main__":
    rd.seed(15)

    # G = tsp.randomGraph(10)
    dataFrameOS = df.readOptimalSolution()
    dirlist = os.listdir(os.path.join(INSTANCES_PATH, 'tsp_data'))
    dataFrame = df.createDataFrame()
    k = 5
    for instance in dirlist:
        G = tsplib.load(os.path.join(INSTANCES_PATH, 'tsp_data', instance)).get_graph(True)
        ot = OneTree(G, trace=True)
        delaunay = delaunayTessellation(G)
        dataFrame = df.insertDataFrame(dataFrameOS, dataFrame, instance, 'Delaunay-Tessellation', 3, delaunay)

        delaunayNearest = delaunayNearestNeighbor(G, ot.original_mat)
        dataFrame = df.insertDataFrame(dataFrameOS, dataFrame, instance, 'Delaunay-NearestNeighbor', 3, delaunayNearest)

        for i in range(k):
            nearest = nearestNeighbor(G, ot.original_mat, (i + 1))
            dataFrame = df.insertDataFrame(dataFrameOS, dataFrame, instance, 'Nearest-Neighbor', (i + 1), nearest)

            ot = OneTree(G, trace=True)
            ub, route = tsp.heuristica2(ot.original_mat, ot.best_pi)
            print('ini ub: ', ub)
            tsp.route_to_graph(route, ot.one_tree)
            alternateEdges = ot.oneTreeModified(ot.original_mat, ot.best_pi, i)
            dataFrame = df.insertDataFrame(dataFrameOS, dataFrame, instance, 'oneTree-Modified', (i + 1),
                                           alternateEdges)

            delaunayOnetree = OneTree(G, trace=True)
            ub, route = tsp.heuristica2(delaunayOnetree.original_mat, delaunayOnetree.best_pi)
            tsp.route_to_graph(route, delaunayOnetree.one_tree)
            delaunayOnetreeEdges = delaunayOnetree.delaunayOneTree(G, delaunayOnetree.original_mat,
                                                                   delaunayOnetree.best_pi)
            dataFrame = df.insertDataFrame(dataFrameOS, dataFrame, instance, 'Delanay-OneTree', 1, delaunayOnetreeEdges)

            print(dataFrame)
            dataFrame.to_csv('dataFrameInstances.csv', index=False)
            teste = nx.Graph()
            teste.add_edges_from(delaunayOnetreeEdges)
            nx.draw(teste, with_labels=True, node_size=500, node_color='lightgreen')
            plt.show()

    # FEITO - porcentagem de arestas dos resultados que estão na solução ótima
    # FEITO - percentual da solução ótima que está no resultado
    # FEITO - fazer o k variar [1,2,3,4,5]
    # criar outro método, combinação de Dalaunay com o onetree modificado, Delaunay com vizinhos próximos
    # pegar 20 instancias
    # FEITO - colocar a porcentagem na tabela - DF

    # ot.plot_one_tree()
    # ot.trace = True
    # ot.branchNBound(ub)

    print("Fim!")
    # plt.show()
    # ot.LAMBDA = 1.1
    # ot.subgradient(ub)
    # ot.subgradient2(ub)
    # ot.subgradient2(ub)