import numpy as np
import tensorflow as tf

def labview_test(intensity_graph):
    [n,m] = np.shape(intensity_graph)
    test_graph = np.ones((n,m))*255
    
    return test_graph
