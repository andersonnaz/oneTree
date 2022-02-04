from itertools import combinations
import networkx as nx
import pandas as pd
import tsplib95 as tsplib
import numpy as np
import os

from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

from one_tree.oneTree import OneTree

INSTANCES_PATH = '../../instances'


def progress(done, total, text: str, end=''):
    x = int(round(40.0 * done / total))
    print(f"\r{text}: |{'█' * x}{'-' * (40 - x)}|", end=end)
    if done == total:
        print()
    pass


def _create_DF():
    """
    Cria um novo dataframe com os campos 'instance', 'method', 'parameter' e 'wdges'
    :return: pandas data frame
    """
    df = pd.DataFrame(
        columns=['instance', 'method', 'parameter', 'edges']
    )
    return df


def _read_opt_sol(path: str):
    """
    Ler o arquivo com a solução ótima da tsplib
    :param path: path para o arquivo
    :return: set com as tuplas (i,j), onde i < j, correspondente as arestas da solução
    """
    tour = tsplib.load(path)
    tour = np.array(tour.tours[0]) - 1
    edges = set([(tour[0], tour[-1])])
    for i in range(1, len(tour)):
        edges.add((tour[i - 1], tour[i]) if tour[i - 1] < tour[i] else (tour[i], tour[i - 1]))
    return edges


def read_opt_dir(path_dir: str):
    """
        Ler todos  arquivo com a solução ótima da tsplib no diretório
        :param path_dir: path para o diretório contendo arquivos de solução ótima
        :return: lista de dicionários {'instance':<nome da instância>, 'method':'opt', 'parameter':None, 'edges':[(i,j)...]}
        """
    lista = []
    file_list = os.listdir(path_dir)
    for i, file_name in enumerate(file_list):
        edges = _read_opt_sol(os.path.join(path_dir, file_name))
        d = {'instance': os.path.basename(file_name).split('.')[0],
             'method': 'opt',
             'parameter': None,
             'edges': edges}
        lista.append(d)
        progress(i + 1, len(file_list), 'Lendo soluções', file_name)
    print()
    return lista


def _delaunayTessellation(G):
    """
    Realiza a triangulação de Delaunay
    :param G: Grafo networkx com campo 'coord' contendo coordenadas dos vertices
    :return: set com as tuplas (i,j), onde i < j, correspondente as arestas da triangulação
    """
    coords = G.nodes.data('coord')
    c = np.array([coord for i, coord in coords])
    d = Delaunay(c)
    edges = set()
    for triangle in d.simplices:
        for e in combinations(triangle, 2):
            edges.add(tuple(sorted(e)))
    return edges


def read_graphs(path_dir: str):
    """
    Ler todos  arquivo de instancias da tsplib no diretório
    :param path_dir: path para o diretório contendo arquivos de isntâncias da tsplib
    :return: dicionários {<nome da instância>: <nx graph>}
    """
    g = {}
    file_list = os.listdir(path_dir)
    for i, file_name in enumerate(file_list):
        g[os.path.basename(file_name).split('.')[0]] = tsplib.load(os.path.join(path_dir, file_name)).get_graph(
            normalize=True)
        progress(i + 1, len(file_list), 'Lendo instâncias', file_name)
        # break
    print()
    return g


def delaunay(path_dir: str):
    """
    Ler todos  arquivo de instancias da tsplib no diretório
    :param graphs: dicionário {<nome>: <graphNX>}
    :return: lista de dicionários {'instance':<nome da instância>, 'method':'delaunay', 'parameter':None, 'edges':[(i,j)...]}
    """
    lista = []
    for i, (name, G) in enumerate(graphs.items()):
        edges = _delaunayTessellation(G)
        d = {'instance': os.path.basename(name).split('.')[0],
             'method': 'delaunay',
             'parameter': None,
             'edges': edges}
        lista.append(d)
        progress(i + 1, len(graphs), 'delaunay', name)
    return lista


def _nearest(G, k: int):
    """
    Seleciona os k vizinhos mais próximo de cada vertice
    :param G: grafo networkx da instância
    :param k: número de vizinhos
    :return:  set com as tuplas (i,j), onde i < j, correspondente as arestas dos k vizinhos mais próximos
    """
    mat = nx.to_numpy_matrix(G)
    edges = set()
    for i in range(len(mat)):
        p = np.asarray(np.argpartition(mat[i], k + 1)).reshape(-1)
        for j in range(k + 1):
            if i != p[j]:
                e = (i, p[j]) if i < p[j] else (p[j], i)
                edges.add(e)

    return edges


def nearest_neigh(graphs: dict, k: int):
    """
    Ler todos  arquivo de instancias da tsplib no diretório
    :param graphs: dicionário {<nome>: <graphNX>}
    :return: lista de dicionários {'instance':<nome da instância>, 'method':'nearest', 'parameter':k, 'edges':[(i,j)...]}
    """
    lista = []
    for i, (name, G) in enumerate(graphs.items()):
        edges = _nearest(G, k)
        d = {'instance': os.path.basename(name).split('.')[0],
             'method': 'nearest',
             'parameter': k,
             'edges': edges}
        lista.append(d)
        progress(i + 1, len(graphs), f'nearest_neigh k= {k}', name)
    return lista


def one_tree(graphs: dict, k: int):
    """
    Ler todos  arquivo de instancias da tsplib no diretório
    :param graphs: path para o diretório contendo arquivos de isntâncias da tsplib
    :return: lista de dicionários {'instance':<nome da instância>, 'method':'one_tree', 'parameter':k, 'edges':[(i,j)...]}
    """
    lista = []

    for i, (name, G) in enumerate(graphs.items()):
        progress(i, len(graphs), f'one_tree k= {k}', name)
        ot = OneTree(G, minL=1e-3, LAMBDA=1.5, iniL=2, trace=False)
        edges = ot.oneTree_alternative(k)
        d = {'instance': name,
             'method': 'on_tree',
             'parameter': k,
             'edges': edges}
        lista.append(d)
        # break
    print()
    return lista


def plot(graph, routes=None, edges=None, clear_edges=True, stop=True, sleep_time=0.01):
    """
    Exibe a instância graficamente

    :param routes: Solução (lista de listas)
    :param edges: lista de arcos (lista de tuplas (i,j) )
    :param clear_edges: limpar o último plot ou não
    :param stop: Parar a execução ou não
    :param sleep_time: Se stop for Falso, tempo de espera em segundos antes de prosseguir
    """
    if clear_edges:
        graph.clear_edges()
    if routes is not None:
        for r in routes:
            if len(r) > 1:
                for i in range(len(r) - 1):
                    graph.add_edge(r[i], r[i + 1])
                graph.add_edge(r[-1], r[0])
    if edges is not None:
        for i, j in edges:
            graph.add_edge(i, j)
    plt.clf()
    color = ['#74BDCB' for i in range(len(graph))]
    color[0] = '#FFA384'
    nx.draw_networkx(graph, graph.nodes.data('coord'), with_labels=True, node_size=120, font_size=8, node_color=color)
    if stop:
        plt.show()
    else:
        plt.draw()
        plt.pause(sleep_time)
    pass


# def insertDataFrame(dataFrameOS, dataFrame, instance, method, parameter, edges):
#     instance = instance[:-4]
#     dataFrameOS = dataFrameOS.set_index('Instance')
#     edgesOS = dataFrameOS.loc[instance, 'Edges']
#     resultIn_os = round(percentage(edges, edgesOS), 2)
#     os_InResult = round(percentage(edgesOS, edges), 2)
#     data = {
#         'Instance': instance,
#         'Method': method,
#         'Parameter': parameter,
#         'Edges in EdgesOS': resultIn_os,
#         'EdgesOS in Edges': os_InResult,
#         'Edges': edges
#     }
#     dataFrame = dataFrame.append(data, ignore_index=True)
#     return dataFrame


# def createDataFrameOS():
#     df = pd.DataFrame(
#         columns=['Instance', 'Edges'],
#     )
#     return df


# def insertDataFrameOS(dataFrame, instance, edges):
#     data = {
#         'Instance': instance,
#         'Edges': edges
#     }
#     dataFrame = dataFrame.append(data, ignore_index=True)
#     return dataFrame


# def read_optimal_sol():
#     # Tour ótimo dos problemas:
#     dirlist = os.listdir(os.path.join(INSTANCES_PATH, 'tsp_opt'))
#     dataFrameOS = createDataFrameOS()
#     for instance in dirlist:
#         tour = tsplib.load(os.path.join(INSTANCES_PATH, 'tsp_opt', instance))
#         tour = np.array(tour.tours[0]) - 1
#         edges = set([(tour[0], tour[-1])])
#         for i in range(1, len(tour)):
#             edges.add((tour[i - 1], tour[i]) if tour[i - 1] < tour[i] else (tour[i], tour[i - 1]))
#         dataFrameOS = insertDataFrameOS(dataFrameOS, instance[:-9], edges)
#
#     dataFrameOS.to_csv('dataFrameOptimalTour.csv', index=False)
#     return dataFrameOS


# def percentage(numerador, denominador):
#     intersect = set(numerador) & set(denominador)
#     return ((len(intersect) * 100) / len(denominador))

if __name__ == "__main__":

    opt = read_opt_dir(os.path.join(INSTANCES_PATH, 'tsp_opt'))
    df = pd.DataFrame(opt)
    df.to_csv('edges_table.csv', index=None)
    #
    instance_path = os.path.join(INSTANCES_PATH, 'tsp_data')
    graphs = read_graphs(instance_path)

    delaunay = delaunay(graphs)
    df = df.append(delaunay)
    df.to_csv('edges_table.csv', index=None)

    for k in range(3, 6):
        nearest = nearest_neigh(graphs, k)
        df = df.append(nearest)
        df.to_csv('edges_table.csv', index=None)

    for k in range(0, 2):
        one_t = one_tree(graphs, k)
        df = df.append(one_t)
        df.to_csv('edges_table.csv', index=None)
        # break

    # só pra teste
    # plot(graph[opt[0]['instance']],edges=opt[0]['edges'])
    # plot(graph[delaunay[0]['instance']], edges=delaunay[0]['edges'])
    # plot(graph[nearest[0]['instance']], edges=nearest[0]['edges'])
    # plot(graphs[one_t[0]['instance']], edges=one_t[0]['edges'])
