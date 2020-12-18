"""
Main Functions
PHY 407 - Gait Distinction via Hip-Knee Coordination
Authors:
        Abdullah Zafar, 999730411
"""
from PHY407_Zafar_Functions_Helper import *
from PHY407_Zafar_Functions_Plot import *
from PHY407_Zafar_Functions_CRP import *
from PHY407_Zafar_Functions_Topology import *
from PHY407_Zafar_Functions_Wasserstein import *
import numpy as np
import time


def continuousrelphase(raw, trial):
    """
    Main function to compute the continuous relative phase from a formatted data array

    Parameters
    ----------
    raw : array[float], size: Nx4
        N - number of observations
        4 - series of data in the following order:
            column 1: Hip angle
            column 2: Hip angular velocity
            column 3: Knee angle
            column 4: Knee angular velocity
    trial : string
        label of the data for presentation purposes

    Returns
    -------
    crp : array[float], Nx4
        continuous relative phase angle for the corresponding columns in raw
    crpdot : array[float], Nx4
        continuous relative phase anglular velocity for the corresponding columns in raw

    """
    print("---- ---- ---- ---- ---- ---- ---- ---- ---- ----")
    print("---- COMPUTING CONTINUOUS RELATIVE PHASE: " + str(trial) + " ----")
    # 1. Process raw data into joint angles/velocities and normalize
    thHN, omHN, thKN, omKN = processphasespace(raw)
    # 2. Compute relative phase differences of hip vs knee phase spaces
    crp = relphasediff(thHN, omHN, thKN, omKN)    
    # 3. Compute first derivative of CRP
    crpdot = np.gradient(crp)
    
    return crp, crpdot

def persistenthomology(x, y, xrange, yrange, landmarks, start, step, end, pltWitComp, trial):
    """
    Main function to compute the persistent homology of a point cloud

    Parameters
    ----------
    x : array[float], size: Nx1
        array of x values in point cloud
    y : array[float], size: Nx1
        array of y values in point cloud
    xrange : list[float], length: 2
        list of minimum/maximum values in x-values: [min(x), max(x)]
    yrange : list[float], length: 2
        list of minimum/maximum values in y-values: [min(y), max(y)]
    landmarks : int
        number of landmark points to use in witness complex construction
    start : float
        smallest epsilon to use in Vietoris-Rips filtration computation
    step : float
        step size for epsilon to use in Vietoris-Rips filtration computation
    end : float
        largest epsilon to use in Vietoris-Rips filtration computation
    pltWitComp : boolean
        True --> plot witness complex
    trial : string
        label of the data for presentation purposes

    Returns
    -------
    intervals : array[floats], size: Kx2
        array of the K homology intervals of the point cloud
            --> each row is an interval
            --> column 1: formation of hole, column 2: closure of hole

    """
    print("---- ---- ---- ---- ---- ---- ---- ---- ---- ----")
    print("---- COMPUTING PERSISTENT HOMOLOGY: " + str(trial) + " ----")
    
    # 1. Store data points as list of tuples and get range of data
    print("1. Normalizing points")
    points = gennormpoints(x,y,xrange,yrange)

    # 2. Approximate point cloud with witness complex
    print("2. Producing Witness Complex")
    pointsL = witnesscomplex(points, landmarks)
    # Plot Witness Complex
    if pltWitComp:
        plotwitcomplex(points, pointsL, D, trial)
    
    # 3. Generate a list of simplices and metric indices from a
    # Vietoris-Rips flitration on the reduced point cloud
    print("3. Creating Vietoris-Rips Filtration")
    list_simplices, list_eps = vrfilt(start, end, step, pointsL)
    print("-> total of " + str(len(list_simplices)) + " simplices in filtration")
    # 4. Create a boundary matrix of the V-R filtration
    print("4. Creating Boundary Matrix")
    tstart = time.perf_counter()
    boundary = createboundarymat(list_simplices)
    # 5. Reduce the boundary matrix
    print("5. Reducing Boundary Matrix")
    boundary_red = reduceboundarymat(boundary)
    texecute = time.perf_counter() - tstart # Execution time
    print("-> boundary matrix reduced in : " +str(texecute) + " sec")
    # 6. Compute the k-largest intervals from reduced boundary matrix
    print("6. Fetching Homology intervals")
    intervals = getintervals(boundary_red, list_eps)

    print("---- ---- ---- ---- ---- ---- ---- ---- ---- ----")
    
    return intervals

def wassersteindist(intervals,k):
    """
    Main function to compute the wasserstein distance between intervals for each trial

    Parameters
    ----------
    intervals : list[array[floats]], length: N
        list of N homology intervals
    k : TYPE
        number of homology intervals to consider for distance computation

    Returns
    -------
    D : matrix[float], size: NxN
        Wasserstein distance matrix

    """
    print("---- COMPUTING WASSERSTEIN DISTANCES ----")
    print("Progress: ", end="")
    # Keep track of progress
    progress = 0
    total = len(intervals)*(len(intervals)-1)/2
    
    # Compute Wasserstein distance matrix
    D = np.zeros((len(intervals),len(intervals)))
    for a in range(len(intervals)):
        for b in range(a+1,len(intervals)):
            # Take the k longest intervals in each set
            B1 = intervals[a][-k:,:]
            B2 = intervals[b][-k:,:]
            # 1. Project intervals onto diagonal
            B1D, B2D = projectdiagonals(B1,B2)
            # 2. Exchange intervals and diagonals
            B1_B2D, B2_B1D = exchangediagonals(B1, B2, B1D, B2D)
            # 3. Create bipartite cost matrix
            Cost = createbipartitematrix(B1_B2D, B2_B1D)
            # 4. Reduce cost matrix with Hungarian Algorithm
            Cost_red = hungarianalgorithm(Cost)
            # 5. Set optimal assignment
            pair = assignpairs(Cost_red)
            # 6. Compute distance
            dist = wassersteindistpairwise(B1_B2D, B2_B1D, pair)
            D[a,b] = dist

            # Update and print progress
            progress += 1
            print("..." + str(round(100*progress/total)) + "%", end="")
            
    D += D.transpose()
    print("")
    
    return D