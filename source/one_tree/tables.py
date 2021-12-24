from itertools import combinations
import networkx as nx
import pandas as pd
import tsplib95 as tsplib
import numpy as np
import os
from scipy.spatial import Delaunay

INSTANCES_PATH = '../../instances'


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
    for file_name in file_list:
        edges = _read_opt_sol(os.path.join(path_dir, file_name))
        d = {'instance': os.path.basename(file_name).split('.')[0],
             'method': 'opt',
             'parameter': None,
             'edges': edges}
        lista.append(d)
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


def delaunay(path_dir: str):
    """
    Ler todos  arquivo de instancias da tsplib no diretório
    :param path_dir: path para o diretório contendo arquivos de isntâncias da tsplib
    :return: lista de dicionários {'instance':<nome da instância>, 'method':'delaunay', 'parameter':None, 'edges':[(i,j)...]}
    """
    lista = []
    file_list = os.listdir(path_dir)
    for file_name in file_list:
        G = tsplib.load(os.path.join(path_dir, file_name)).get_graph(normalize=True)
        edges = _delaunayTessellation(G)
        d = {'instance': os.path.basename(file_name).split('.')[0],
             'method': 'delaunay',
             'parameter': None,
             'edges': edges}
        lista.append(d)
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


def nearest_neigh(path_dir: str, k: int):
    """
    Ler todos  arquivo de instancias da tsplib no diretório
    :param path_dir: path para o diretório contendo arquivos de isntâncias da tsplib
    :return: lista de dicionários {'instance':<nome da instância>, 'method':'nearest', 'parameter':k, 'edges':[(i,j)...]}
    """
    lista = []
    file_list = os.listdir(path_dir)
    for file_name in file_list:
        G = tsplib.load(os.path.join(path_dir, file_name)).get_graph(normalize=True)
        edges = _nearest(G, k)
        d = {'instance': os.path.basename(file_name).split('.')[0],
             'method': 'nearest',
             'parameter': k,
             'edges': edges}
        lista.append(d)
    return lista

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
