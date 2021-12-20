from networkx.algorithms.operators.binary import intersection
import pandas as pd
import tsplib95 as tsplib
import numpy as np

# from numba import jit
from assignment import assigment

import six
import sys
import os

from oneTreeV02.oneTree import INSTANCES_PATH

sys.modules['sklearn.externals.six'] = six

def createDataFrame():
    df = pd.DataFrame(
            columns=['Instance', 'Method', 'Parameter','Edges in EdgesOS', 'EdgesOS in Edges', 'Edges'],
        )
    return df

def insertDataFrame(dataFrameOS, dataFrame, instance, method, parameter, edges):
        instance = instance[:-4]
        dataFrameOS = dataFrameOS.set_index('Instance')
        edgesOS = dataFrameOS.loc[instance, 'Edges']
        resultIn_os = round(percentage(edges, edgesOS), 2)
        os_InResult = round(percentage(edgesOS, edges), 2)
        data = {
            'Instance': instance,
            'Method': method,
            'Parameter': parameter,
            'Edges in EdgesOS': resultIn_os,
            'EdgesOS in Edges': os_InResult,
            'Edges': edges
        }
        dataFrame = dataFrame.append(data, ignore_index=True)
        return dataFrame

def createDataFrameOS():
    df = pd.DataFrame(
            columns=['Instance', 'Edges'],
        )
    return df

def insertDataFrameOS(dataFrame, instance, edges):
    data = {
            'Instance': instance,
            'Edges': edges
        }
    dataFrame = dataFrame.append(data, ignore_index=True)
    return dataFrame

def readOptimalSolution():
    # Tour ótimo dos problemas:
    dirlist = os.listdir(os.path.join(INSTANCES_PATH, 'tsp_opt'))
    dataFrameOS = createDataFrameOS()
    for instance in dirlist:
        lista = []
        edges = set()
        tour = tsplib.load(os.path.join(INSTANCES_PATH, 'tsp_opt', instance))
        lista = list(np.array(tour.tours[0]) - 1)
        edges = set([(lista[-1], lista[0])])
        for i in range(len(lista) - 1):
            j = i + 1
            edges.add((lista[i], lista[j]))
        dataFrameOS = insertDataFrameOS(dataFrameOS, instance[:-9], edges)
        dataFrameOS.to_csv('dataFrameOptimalTour.csv', index=False)
        # print(dataFrameOS)
    return dataFrameOS

def percentage(numerador, denominador):
    intersect = set(numerador) & set(denominador)
    return ((len(intersect)*100)/len(denominador))