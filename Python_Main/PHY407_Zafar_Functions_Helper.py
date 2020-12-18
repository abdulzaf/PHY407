"""
Helper Functions - General
PHY 407 - Gait Distinction via Hip-Knee Coordination
Authors:
        Abdullah Zafar, 999730411
"""

import numpy as np

def norm(x,y):
    """
    Compute the Euclidean 2-norm of a pair of points

    Parameters
    ----------
    x : tuple[float], length: 2
        first point
    y : tuple[float], length: 2
        second point

    Returns
    -------
    distance, float
        Euclidean distance between x and y

    """
    distance = np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    
    return distance

def normalize(x,minmax):
    """
    Linear normalization of a set of values to a min and max, where min is
    mapped to -1 and max is mapped to 1

    Parameters
    ----------
    x : array[float]
        array of values to normalize
    minmax : list[float], length: 2
        list of min, max reference points: [min, max]

    Returns
    -------
    xN : array[float]
        array of values linearly scaled into the range [-1,1]

    """
    # Bring into range [0,1]
    xN = (x - minmax[0]) / (minmax[1] - minmax[0])
    # Scale to range [-1,1]
    xN = 2*xN - 1
    
    return xN

def gennormpoints(x,y,xrange,yrange):
    """
    Given a set of x/y coordinates, produce a list of normalized tuples (x,y)

    Parameters
    ----------
    x : array[float]
        array of x-values
    y : array[float]
        array of y-values
    xrange : list[float], length: 2
        list of minimum/maximum values in x-values: [min(x), max(x)]
    yrange : list[float], length: 2
        list of minimum/maximum values in y-values: [min(y), max(y)]

    Returns
    -------
    points : list[tuples], length: N
        list of N points in the point cloud, each point expressed as a tuple (x,y)

    """
    # Normalize points
    xN = normalize(x,xrange)
    yN = normalize(y,yrange)
    # Create list of tuples (x[i],y[i])
    points = []
    for i in range(len(x)):
        points.append((xN[i],yN[i]))
        
    return points