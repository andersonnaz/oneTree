import numpy as np
# Biblioteca para trabalhar com grafos
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import random as rd
import pandas as pd

# from numba import jit
from assignment import assigment

import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose

def createDataFrame():
    data = {
        'Instance': 'default',
        'Method': 'default',
        'Parameter': 0,
        'Edges': 'default'
    }
    df = pd.DataFrame(data)
    return df