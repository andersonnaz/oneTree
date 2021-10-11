# Biblioteca para trabalhar com grafos
import networkx as nx

from scipy.sparse.csgraph import minimum_spanning_tree as mst

# Biblioteca para plotar os grafos
import matplotlib.pyplot as plt

import tsplib95 as tsplib

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from scipy.optimize import linear_sum_assignment as ap

from assignment import assigment

import numpy as np
import random as rd

import tsp


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

        if (self.ub - lb+fixedCost)/self.ub > .01:
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
        return

    def plot_one_tree(self):
        plt.clf()
        nx.draw_networkx(self.one_tree, self.one_tree.nodes.data('coord'), with_labels=True, node_size=100,
                         font_size=8)
        plt.draw()
        plt.pause(1)

    def branchNBound(self, UB):
        self.ub = UB
        self.__branch([], [])


if __name__ == "__main__":
    rd.seed(7)
    G = tsplib.load('./instances/tsp_data/berlin52.tsp').get_graph(True)
    # G = tsp.randomGraph(100)
    ot = OneTree(G, trace=True)
    ub, route = tsp.heuristica2(ot.original_mat,ot.best_pi)
    print('ini ub: ', ub)
    tsp.route_to_graph(route,ot.one_tree)
    ot.plot_one_tree()
    # ot.trace = True
    ot.branchNBound(ub)

    print("Fim!")
    plt.show()
    # ot.LAMBDA = 1.1
    # ot.subgradient(ub)
    # ot.subgradient2(ub)
    # ot.subgradient2(ub)
