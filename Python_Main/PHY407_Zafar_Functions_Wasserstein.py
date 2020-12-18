"""
Helper Functions - Wasserstein Distance Computation
PHY 407 - Gait Distinction via Hip-Knee Coordination
Authors:
        Abdullah Zafar, 999730411
"""

from PHY407_Zafar_Functions_Helper import norm
import numpy as np
from itertools import permutations

def projectdiagonals(B1, B2):
    """
    Given two sets of homology intervals, project the interval coordinates onto
    the diagonal

    Parameters
    ----------
    B1 : array[float], size: Kx2
        array of the K homology intervals
            --> each row is an interval
            --> column 1: formation of hole, column 2: closure of hole
    B2 : array[float], size: Kx2
        array of the K homology intervals
            --> each row is an interval
            --> column 1: formation of hole, column 2: closure of hole

    Returns
    -------
    B1D : array[float], size: Kx2
        array of the K homology intervals projected onto the diagonal
    B2D : array[float], size: Kx2
        array of the K homology intervals projected onto the diagonal

    """
    B1D = np.vstack((B1[:,1],B1[:,1])).transpose()
    B2D = np.vstack((B2[:,1],B2[:,1])).transpose()
    
    return B1D, B2D

def exchangediagonals(B1, B2, B1D, B2D):
    """
    Given two sets of homology intervals and their projected diagonals, appends
    the diagonals of one set onto the other

    Parameters
    ----------
    B1 : array[float], size: Kx2
        array of the K homology intervals
    B2 : array[float], size: Kx2
        array of the K homology intervals
    B1D : array[float], size: Kx2
        array of the K homology intervals projected onto the diagonal
    B2D : array[float], size: Kx2
        array of the K homology intervals projected onto the diagonal

    Returns
    -------
    B1_B2D : array[float], size: (K+K)x2
        array of K homology intervals + K projected intervals
    B2_B1D : array[float], size: (K+K)x2
        array of K homology intervals + K projected intervals

    """
    B1_B2D = np.vstack((B1,B2D))
    B2_B1D = np.vstack((B2,B1D))
    
    return B1_B2D, B2_B1D

def createbipartitematrix(set1, set2):
    """
    Given an augmented set of homology intervals, create a bipartite distance
    matrix, relating the Euclidean distance from each interval in the first set
    to each interval in the second set

    Parameters
    ----------
    set1 : array[float], size: Kx2
        array of the K homology intervals
    set2 : array[float], size: Kx2
        array of the K homology intervals

    Returns
    -------
    D : matrix[float], size: KxK
        bipartite distance matrix between each interval in set1 to set2

    """
    n = len(set1)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = norm(tuple(set1[i,:]),tuple(set2[j,:]))
            
    return D

def hungarianalgorithm(D):
    """
    Implementation of the Hungarian algorithm for solving the linear assignment
    problem given by cost matrix D

    Parameters
    ----------
    D : matrix[float], size: KxK
        square, bipartite cost matrix representing linear assignment problem

    Returns
    -------
    D : matrix[float], size: KxK
        reduced cost matrix containing solution to linear assignment problem

    """
    n = len(D)
    # 1. Remove min value from each row
    for r in range(n):
        D[r,:] -= min(D[r,:])
        
    # 2. Remove min value from each col
    for c in range(n):
        D[:,c] -= min(D[:,c])
        
    # 3. Keep trying to cover 0s with minimal # of lines until: # of lines = n
    total_cover = 0 # Initialize a count to keep track of covering lines
    while (total_cover<n):
        C = np.zeros(D.shape) # Matrix to keep track of covers
        Z = np.zeros(D.shape) # Matrix to keep track of zero/non-zero values in D
        Z[D!=0] = 1
        
        # Loop to cover 0s in D with lines
        while (np.any(Z==0)):
            # Get sums of rows/cols
            ZR = [sum(Z[r,:]) for r in range(n)]
            ZC = [sum(Z[:,c]) for c in range(n)]
            # Find index of min col/row
            idR = np.argmin(ZR)
            idC = np.argmin(ZC)
            # Cover the row/column with the most zeros
            if ZR[idR]<=ZC[idC]:
                # print(ZR[idR])
                Z[idR,:] = 1
                C[idR,:] = 1
                total_cover += 1
            else:
                # print(ZC[idC])
                Z[:,idC] = 1
                C[:,idC] = 1
                total_cover += 1

        # If the minimal covering lines < n        
        if total_cover<n:
            # Reset the cover count
            total_cover = 0
            
            # Find the smallest uncovered entry
            small_unc = min(D[C==0])
            # a) Subtract smallest uncovered entry from all rows
            row_cov = C.sum(axis=1)<n
            for r in range(n):
                if row_cov[r]:
                    D[r,:] -= small_unc
            
            # b) Add smallest entry to all covered columns
            col_cov = C.sum(axis=0)==n
            for c in range(n):
                if col_cov[c]:
                    D[:,c] += small_unc

    return D

def assignpairs(D):
    """
    Given a reduced cost matrix, find the assignment of rows to columns solving
    the linear assignment problem

    Parameters
    ----------
    D : matrix[float], size: KxK
        reduced cost matrix

    Returns
    -------
    A : array[int], K
        array of column indices which are paired with the corresponding row index
         --> A[0] is the column which is paired with the 0-th row

    """
    D1 = D.copy()
    n = len(D)
    
    # Create matrix to keep track of zeros in D
    Z = np.zeros(D.shape)
    Z[D1!=0] = 1
    ZR = np.array([n-sum(Z[r,:]) for r in range(n)])
    
    # Create array to store pair assignment for each row
    A = np.zeros((n), dtype=int)
    
    # Keep assigning pairs until there are no more pairs to assign
    while (np.any(Z==0)):
        # Prioritize pairing rows with the fewest number of 0s first:
        min_zeros = np.min(ZR[np.nonzero(ZR)])
        # Get first row with fewest zeros
        id_row = min(np.where(ZR==min_zeros)[0])
        # Get id of first zero column in the row
        id_zero = np.where(Z[id_row,:]==0)[0]
        A[id_row] = id_zero[0]
        # Remove the pair option from D
        D1[:,id_zero[0]] = 1
        D1[id_row,:] = 1
        
        
        # Update the matrix keeping track of zeros
        Z[D1!=0] = 1
        ZR = np.array([n-sum(Z[r,:]) for r in range(n)])

    return A

def wassersteindistpairwise(set1, set2, pair):
    """
    Compute the wasserstein distance between two sets of homology intervals,
    given the optimal pairing assignment between them

    Parameters
    ----------
    set1 : array[float], size: Kx2
        array of the K homology intervals
    set2 : array[float], size: Kx2
        array of the K homology intervals
    pair : array[int], K
        array of column indices which are paired with the corresponding row index
         --> A[0] is the column which is paired with the 0-th row

    Returns
    -------
    distance : float
        Wasserstein distance between set1 and set2

    """
    distance = 0
    for i in range(len(set1)):
        distAB = norm(tuple(set1[i,:]),tuple(set2[pair[i],:]))
        distance += distAB
        
    return distance
