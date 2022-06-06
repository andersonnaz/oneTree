# Biblioteca para trabalhar com grafos
import networkx as nx

from scipy.sparse.csgraph import minimum_spanning_tree as mst

# Biblioteca para plotar os grafos
import matplotlib.pyplot as plt

import tsplib95 as tsplib

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from scipy.optimize import linear_sum_assignment as ap

from one_tree.assignment import assignment



import numpy as np
import random as rd

import one_tree.tsp


class OneTree():
    def __init__(self, graph, LAMBDA=1.5, minL=1e-6, iniL=2, trace=False):
        assert isinstance(graph, nx.classes.graph.Graph), "tipo incorreto para graph"
        self.LAMBDA = LAMBDA
        self.minL = minL
        self.iniL = iniL
        self.trace = trace
        self.G = graph
        self.mat = nx.to_numpy_matrix(self.G)
        self.N = len(self.mat)
        for i in range(self.N):
            self.mat[i, i] = np.inf
        self.matb = np.matrix(self.mat)
        self.pi = np.zeros(self.N)
        self.bestPi = np.zeros(self.N)
        self.oneT = nx.Graph.copy(self.G)
        self.Gm1 = nx.Graph.copy(self.G)
        self.Gm1.remove_node(self.N - 1)

    def oneTree(self):
        n = self.N - 1
        for i in range(n):
            for j in range(i):
                self.Gm1.edges[i, j]['weight'] = self.Gm1.edges[j, i]['weight'] = \
                    self.matb[i, j] + self.pi[i] + self.pi[j]
        # gerando a árvore com o restando dos nós.
        spaningTree = nx.minimum_spanning_tree(self.Gm1, algorithm="kruskal", weight='weight')
        # nx.draw_networkx(spaningTree, spaningTree.nodes.data('pos'), with_labels=True, node_size=200, font_size=8)
        # localizar as menores arestas que conectam o nó removido na matriz de distâncias
        vet = np.zeros(n)
        for i in range(n):
            vet[i] = self.matb[n, i] + self.pi[i] + self.pi[n]
        a, b = np.argpartition(vet, 2)[:2]
        # inserindo o nó removido
        self.oneT.clear_edges()
        self.oneT.add_edges_from(spaningTree.edges)
        self.oneT.add_edge(n, a)
        self.oneT.add_edge(n, b)
        return

    def costOneT(self, d):
        cost = 0
        for e in self.oneT.edges:
            cost += self.matb[e]
        return cost

    def subgradient2(self, ub):
        dcost, x, v, u = assignment(self.matb)
        for i in range(self.N):
            self.pi[i] = -0.5 * (v[i] + u[i])

        is_ham = False
        np.copyto(self.bestPi, self.pi)
        L = self.iniL
        self.oneTree()
        d = np.array(self.oneT.degree())[:, 1]
        maxW = w = self.costOneT(d) + np.dot(d - 2, self.pi)
        if self.trace:
            print("ini w ", w)
        ite = 1
        while (L > self.minL and ub > maxW):
            if np.all(d == 2):
                if self.trace:
                    print("Ciclo Hamiltoniano encontrado.")
                is_ham = True
                break
            if ub + 1e-6 < maxW:
                if self.trace:
                    print("UB atingido.")
                break
            # subgradiente
            g = d - 2
            # passo
            t = (ub - w) / (np.linalg.norm(g) ** 2)
            a = 0
            b = 2.0
            while L > self.minL:
                ite += 1
                pi = self.bestPi + L * t * g
                self.oneTree()
                d = np.array(self.oneT.degree())[:, 1]
                w = self.costOneT(d) + np.dot(d - 2, self.pi)
                if w > maxW + 1e-6:
                    np.copyto(self.bestPi, pi)
                    maxW = w
                    L = (a + b) / 2
                    a = L
                    break
                else:
                    L = (a + b) / 2
                    b = L
            if self.trace:
                print("%-6d UBgap:%7.2lf%% \t|g|:%.2lf  \tL:%.9lf" % (ite, 100 * maxW / ub, np.linalg.norm(g), L))

        np.copyto(self.pi, self.bestPi)
        self.oneTree()
        return maxW, is_ham



    def subgradient(self, ub, ini_pi=None):
        if ini_pi is None:
            dcost, x, v, u = assigment(self.matb)
            for i in range(self.N):
                self.pi[i] = -0.5 * (v[i] + u[i])
        else:
            np.copyto(self.pi, ini_pi)
        is_ham = False
        np.copyto(self.bestPi, self.pi)
        L = self.iniL
        self.oneTree()
        d = np.array(self.oneT.degree())[:, 1]
        maxW = w = self.costOneT(d) + np.dot(d - 2, self.pi)
        if self.trace:
            print("ini w ", w)
        ite = 1
        while (L > self.minL and ub > maxW):
            if np.all(d == 2):
                if self.trace:
                    print("Ciclo Hamiltoniano encontrado.")
                    print("Ciclo Hamiltoniano encontrado.")
                is_ham = True
                break
            if ub + 1e-6 < maxW:
                if self.trace:
                    print("UB atingido.")
                break
            # subgradiente
            g = d - 2
            # passo
            t = (ub - w) / (np.linalg.norm(g) ** 2)
            # self.pi = self.bestPi
            np.copyto(self.pi, self.bestPi)
            last_w = w
            while L > self.minL:
                ite += 1
                self.pi = self.pi + L * t * g
                self.oneTree()
                d = np.array(self.oneT.degree())[:, 1]
                g = d - 2
                t = (ub - w) / (np.linalg.norm(g) ** 2)
                w = self.costOneT(d) + np.dot(d - 2, self.pi)
                if w > maxW + 1e-6:
                    np.copyto(self.bestPi, self.pi)
                    maxW = w
                    break
                elif w < last_w:
                    L /= self.LAMBDA
                last_w = w
            if self.trace:
                print("%-6d UBgap:%7.2lf%% \t|g|:%.2lf  \tL:%.9lf" % (ite, 100 * maxW / ub, np.linalg.norm(g), L))

        np.copyto(self.pi, self.bestPi)
        self.oneTree()
        return maxW, is_ham

    def subgradient_bak(self, ub):
        # dcost, x, v, u = assigment(self.matb)
        # for i in range(self.N):
        #     self.pi[i] = -0.5 * (v[i] + u[i])

        is_ham = False
        np.copyto(self.bestPi, self.pi)
        L = self.iniL
        self.oneTree()
        d = np.array(self.oneT.degree())[:, 1]
        maxW = w = self.costOneT(d) + np.dot(d - 2, self.pi)
        if self.trace:
            print("ini w ", w)
        ite = 1
        while (L > self.minL and ub > maxW):
            if np.all(d == 2):
                if self.trace:
                    print("Ciclo Hamiltoniano encontrado.")
                is_ham = True
                break
            if ub + 1e-6 < maxW:
                if self.trace:
                    print("UB atingido.")
                break
            # subgradiente
            g = d - 2
            # passo
            t = (ub - w) / (np.linalg.norm(g) ** 2)
            while L > self.minL:
                ite += 1
                self.pi = self.bestPi + L * t * g
                self.oneTree()
                d = np.array(self.oneT.degree())[:, 1]
                w = self.costOneT(d) + np.dot(d - 2, self.pi)
                if w > maxW + 1e-6:
                    np.copyto(self.bestPi, self.pi)
                    maxW = w
                    break
                else:
                    L /= self.LAMBDA
            if self.trace:
                print("%-6d UBgap:%7.2lf%% \t|g|:%.2lf  \tL:%.9lf" % (ite, 100 * maxW / ub, np.linalg.norm(g), L))

        np.copyto(self.pi, self.bestPi)
        self.oneTree()
        return maxW, is_ham

    def branch(self, X, Y, l=0):
        ec = []
        np.copyto(self.matb, self.mat)
        fixedCost = 0
        for i, j in X:
            self.matb[i, j] = self.matb[j, i] = 0
            fixedCost += self.mat[i, j]
        for i, j in Y:
            self.matb[i, j] = self.matb[j, i] = np.inf

        lb, isHam = self.subgradient(self.ub - fixedCost)
        print(lb + fixedCost, l, len(X), '-', len(Y))
        if lb + fixedCost >= self.ub:
            return
        elif isHam:  # novo incumbent
            self.ub = lb + fixedCost
            print('novo incumbent ', self.ub)
            plt.clf()
            nx.draw_networkx(self.oneT, self.oneT.nodes.data('coord'), with_labels=True, node_size=100, font_size=8)
            plt.draw()
            plt.pause(1)
            return

        for e in self.oneT.edges:
            if e in X or e in Y:
                continue
            i, j = e
            self.matb[i, j] = self.matb[j, i] = np.inf
            self.oneTree()
            d = np.array(self.oneT.degree())[:, 1]
            w = self.costOneT(d) + np.dot(d - 2, self.pi)
            ec.append((w + fixedCost, e))
            self.matb[i, j] = self.matb[j, i] = self.mat[i, j]
        ec = sorted(ec, reverse=True)
        for i in range(0, len(ec)):
            if i > 0:
                X.append(ec[i - 1][1])
            Y.append(ec[i][1])
            self.branch(list(X), Y, l + 1)
            Y.remove(ec[i][1])
        return

    def branchNBound(self, UB):
        self.ub = UB
        self.branch([], [])

    def oneTree_alternative(self, k=2):
        mat = self.mat
        matb = self.matb
        ub, path = tsp.NearestNeighbor(mat)
        path, ub = tsp.VND(path, mat)
        self.subgradient(ub)
        edges = set(self.oneT.edges)
        if k == 0:
            return edges
        alternativas = set()
        for i, e in enumerate(edges):
            np.copyto(matb, mat)
            pivot = e
            matb[pivot] = matb[pivot[::-1]] = np.inf
            for j in range(k):
                print(f"edge: {i + 1}/{len(edges)} k:{j + 1}/{k}")
                self.subgradient(ub, self.bestPi)
                novos = set(self.oneT.edges) - edges
                for n in novos:
                    alternativas.add(n)
                    matb[n] = matb[n[::-1]] = np.inf
        return edges.union(alternativas)

# if __name__ == "__main__":
#     rd.seed(7)
#     # G = tsplib.load('att48.tsp').get_graph(True)
#     G = tsp.randomGraph(30)
#
#     # ub, path = tsp.NearestNeighbor(ot.mat)
#     # path, ub = tsp.VND(path, ot.mat)
#     # ub = tsp.cost(path, ot.mat)
#     # print('ini ub: ', ub)
#     # ot.branchNBound(ub)
