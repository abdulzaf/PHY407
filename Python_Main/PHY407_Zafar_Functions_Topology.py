"""
Helper Functions - Persistent Homology Computation
PHY 407 - Gait Distinction via Hip-Knee Coordination
Authors:
        Abdullah Zafar, 999730411
"""

from PHY407_Zafar_Functions_Helper import norm
import numpy as np
from itertools import combinations
from random import seed, randint

def witnesscomplex(points, nland):
    """
    Downsample a point cloud using a set of landmark points

    Parameters
    ----------
    points : list[tuples], length: N
        list of N points in the point cloud, each point expressed as a tuple (x,y)
    nland : int
        number of landmark points to construct witness complex

    Returns
    -------
    pointsL : list[tuples], length: nland
        list of nland points, each point expressed as a tuple (x,y)

    """
    npoints = len(points)
    
    # Choose landmarks
    L = []
    # First point
    seed(1)
    L.append(randint(0,npoints-1))
    # max-min algorithm
    for i in range(nland-1):
        # Get min distances to landmarks
        min_dist = []
        for p in range(npoints):
            # If point is not already a landmark
            if p not in L:
                # Get distances to all landmarks
                dist_p = [norm(points[p], points[x]) for x in L]
                # Store minimum distance to all landmarks
                min_dist.append(min(dist_p))
            # Otherwise pad the list of minimum distances with -1
            else:
                min_dist.append(-1)
            
        # Find and store index of max-min dist:
        l = np.argmax(min_dist)
        L.append(l)
    
    pointsL = [points[l] for l in L]
    
    return pointsL

def vrfilt(start, end, step, points):
    """
    Generate the Vietoris-Rips filtration of a 2D point cloud given resolution parameters

    Parameters
    ----------
    start : float
        smallest epsilon to use in Vietoris-Rips filtration computation
    end : float
        largest epsilon to use in Vietoris-Rips filtration computation
    step : float
        step size for epsilon to use in Vietoris-Rips filtration computation
    points : list[tuples], length: N
        list of N points in the point cloud, each point expressed as a tuple (x,y)

    Returns
    -------
    simp_list : list[tuples], length: K
        ordered list of all the simplices in the filtration as tuples of points
        1. simplices are ordered by entry into filtration
        2. simplices are ordered by size
    e_list : list[float], length: K
        dual list to simp_list, keeping track of the entry points of simplices
        into the filtration

    """
    # Initialize parameters
    epsilon = np.arange(start,end+step,step) # range of epsilons to create filtration over
    simp_list = [] # list to hold simplices in filtration
    e_list = [] # list to hold epsilon values at which simplices enter filtration

    # Loop over epsilon to create filtration
    for e in epsilon:
        # Find all edges which are < e apart
        vr1_e = [(x,y) for (x,y) in combinations(points, 2) if norm(x,y) < e]
        # Find all triangles which are < e apart
        vr2_e = [(x,y,z) for (x,y,z) in combinations(points, 3)
                if norm(x,y) < e and norm(x,z) < e and norm(y,z) < e]
        
        # Store edges into filtration
        for simp in vr1_e:
            if not simp in simp_list:
                simp_list.append(simp)
                e_list.append(e)
        # Store triangles into filtration
        for simp in vr2_e:
            if not simp in simp_list:
                simp_list.append(simp)
                e_list.append(e)
    
    return simp_list, e_list


def createboundarymat(simplices):
    """
    Given an ordered list of simplices in a filtration, generate the boundary matrix

    Parameters
    ----------
    simplices : list[tuples], length: K
        ordered list of all the simplices in the filtration as tuples of points
        1. simplices are ordered by entry into filtration
        2. simplices are ordered by size

    Returns
    -------
    delta : matrix[int], size: KxK
        boundary matrix of the filtration
            --> entries are 1 if simplex_i is a face of simplex_j
                --> i/j are row/column indices respectively and i<j
            --> entries are 0 otherwise

    """
    # Initialize a zero matrix for the boundary
    n = len(simplices)
    delta = np.zeros((n,n))
    
    # Iterate of list of simplices
    for i in range(n):
        simp_i = simplices[i]
        for j in range(i,n):
            simp_j = simplices[j]
            # Check if simp_i is a face of simp_j
            if (simp_i[0] in simp_j) and (simp_i[1] in simp_j) and i!=j:
                delta[i,j] = 1
                
    return delta

def getpivotindices(A):
    """
    Get the pivot indices for each column of a matrix A

    Parameters
    ----------
    A : matrix[float], size: NxN
        matrix to find pivots for

    Returns
    -------
    pivot_id : list[int], length: N
        list of pivot indices for each column
            --> if column has no pivots, pivot index is -1

    """
    # Initialize a list to hold pivot indices
    pivot_id = []
    
    # Loop over columns of A
    for j in range(len(A)):
        # If A has a pivot, store its index
        if sum(A[:,j])>0:
            p_idx = np.where(A[:,j])[0][-1]
        # Otherwise store the pivot index as -1
        else:
            p_idx = -1
        pivot_id.append(p_idx)
    
    return pivot_id

def reduceboundarymat(delta):
    """
    Implementation of the standard algorithm to reduce a boundary matrix

    Parameters
    ----------
    delta : matrix[int], size: NxN
        boundary matrix for a filtration of simplicial complexes

    Returns
    -------
    delta_r : matrix[int], size: NxN
        reduced boundary matrix for a filtration of simplicial complexes

    """
    # Create a copy of the boundary matrix
    delta_r = delta.copy()
    # Get pivot indices of the boundary matrix
    pivot_rows = getpivotindices(delta_r)
    # Iterate over the columns of the boundary matrix, from left to right
    for j in range(len(delta_r)):
        pivot_fixed = False
        while pivot_fixed==False:
            # If column j has no pivot, leave it
            if pivot_rows[j]==-1:
                pivot_fixed = True
            else:
                cols = np.where(pivot_rows==pivot_rows[j])[0]
                # If the pivot in column j is not the pivot in a preceding column, leave it
                if all(i >= j for i in cols):
                    pivot_fixed = True
                # Otherwise add the preceding column with same pivot index to column j
                else:
                    col_swap = cols[0]
                    # Add columns and return to binary
                    delta_r[:,j] += delta_r[:,col_swap]
                    delta_r[delta_r==2] = 0
                    # Re-compute pivots
                    pivot_rows = getpivotindices(delta_r)
                    
    return delta_r
                    
def getintervals(delta, epsilon):
    """
    Given a reduced boundary matrix and list of indices at which simplices
    are "born" into a filtration, returns the homology intervals of the filtration

    Parameters
    ----------
    delta : matrix[int], size: NxN
        reduced boundary matrix for a filtration of simplicial complexes
    epsilon : list[float], length: K
        dual list to simp_list, keeping track of the entry points of simplices
        into the filtration

    Returns
    -------
    intervals : array[float], size: Kx2
        array of the K homology intervals
            --> each row is an interval
            --> column 1: formation of hole, column 2: closure of hole

    """
    # Initialize lists to store homology intervals and lifespans
    intervals = []
    lifespan = []
    # Pair each simplex in the filtration with the simplex at its pivot index in delta
    pivot_pairs = np.array([np.arange(0,len(delta),1), getpivotindices(delta)])
    # Iterate over simplices
    for i in range(len(delta)):
        # If a pivot exists, a hole is formed on the entry of the simplex of the
        # pivot row, and is closed on the entry of the simplex of the pivot column
        if pivot_pairs[1,i]>-1:
            birth_idx = epsilon[pivot_pairs[1,i]]
            death_idx = epsilon[pivot_pairs[0,i]]

            # If the hole has a lifespan, add it to the list of intervals
            if death_idx-birth_idx>0:
                intervals.append((birth_idx, death_idx))
                lifespan.append(death_idx-birth_idx)
            
    # Sort the intervals by lifespan length
    int_id = np.array(lifespan).argsort()
    intervals = np.array([intervals[i] for i in int_id])
    
    return intervals