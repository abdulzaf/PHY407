"""
Helper Functions - Continuous Relative Phase Computation
PHY 407 - Gait Distinction via Hip-Knee Coordination
Authors:
        Abdullah Zafar, 999730411
"""

from PHY407_Zafar_Functions_Helper import norm
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 12})

def plotcrp(crp, data_labels):
    """
    Function to plot continuous relative phase over gait cycle % for each trial

    Parameters
    ----------
    crp : list[array], size: Nx1
        list of arrays holding CRP angle values for different trials
    data_labels : string
        list of data labels for each trial

    Returns
    -------
    None.

    """
    plt.figure()
    for i in range(len(crp)):
        plt.title("CRP Hip-Knee")
        plt.plot(100*np.arange(0,len(crp[i]),1)/len(crp[i]),crp[i],label=data_labels[i])
        plt.xlabel("% Gait Cycle")
        plt.ylabel("CRP (degrees)")
        plt.legend()
        
        
def plotcrpphase(crp, crpdot, data_labels):
    """
    Function to plot continuous relative phase space for each trial

    Parameters
    ----------
    crp : list[array], size: Nx1
        list of arrays holding CRP angle values for different trials
    crpdot : list[array], size: Nx1
        list of arrays holding CRP angular velocity values for different trials
    data_labels : string
        list of data labels for each trial

    Returns
    -------
    None.

    """
    plt.figure()
    for i in range(len(crp)):
        plt.title("CRP Phase Space")
        plt.plot(crp[i],crpdot[i],label=data_labels[i])
        plt.xlabel("CRP (degrees)")
        plt.ylabel("CRP Velocity (degrees/sec)")
        plt.legend()

def plotpersistencediagram(start, end, intervals, k, data_labels):
    """
    Plots the persistence diagram for a set of homology intervals for each trial

    Parameters
    ----------
    start : float
        minimum x-value to plot
    end : float
        maximum x-value to plot
    intervals : list[array], length: N
        list of N arrays holding the homology intervals for each trial
    k : int
        number of intervals to plot
    data_labels : string
        list of data labels for each trial

    Returns
    -------
    None.

    """
    plt.figure()
    plt.title("Persistence Diagram")
    plt.plot([start,end],[start,end], 'k--')
    for i in range(len(intervals)):
        interval = intervals[i]
        plt.plot(interval[-k:,0], interval[-k:,1],'o', markersize=12, label=data_labels[i])
        
    plt.xlabel("Birth (epsilon)")
    plt.ylabel("Death (epsilon)")
    plt.legend()
    plt.axis("equal")
    plt.grid()

def plotmatrix(M, data_labels, title):
    """
    Function to plot a data matrix
    Adapted from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

    Parameters
    ----------
    M : matrix[float], size: NxN
        matrix to plot
    data_labels : string
        list of data labels for each trial
    title : string
        title to display

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    im = ax.imshow(M)
    ax.set_title(title)
    
    # Display ticks
    ax.set_xticks(np.arange(len(data_labels)))
    ax.set_yticks(np.arange(len(data_labels)))
    
    # Set tick labels to data labels
    ax.set_xticklabels(data_labels)
    ax.set_yticklabels(data_labels)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(M)):
        for j in range(len(M)):
            text = ax.text(j, i, round(M[i, j]*100)/100,
                           ha="center", va="center", color="w")
            
    fig.colorbar(im)
    ax.axis('equal')
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    

def plotwitcomplex(points, pointsL, data_labels):
    """
    Function to plot a witness complex given a point cloud and set of landmarks

    Parameters
    ----------
    points : list[tuples], length: N
        list of N points in the point cloud, each point expressed as a tuple (x,y)
    pointsL : list[tuples], length: n
        list of n landmark points to construct witness complex
    data_labels : string
        list of data labels for each trial

    Returns
    -------
    None.

    """
    # #% Create Witness-Complex Skeleton
    # Create landmark-witness distance matrix:
    D = np.zeros((len(pointsL), len(points)))
    for i in range(len(pointsL)):
        for j in range(len(points)):
            D[i,j] = norm(pointsL[i], points[j])
    
    # Find vertices of witness edges
    # Iterate over columns of D
    pointsE = []
    for j in range(len(points)):
        min_id = D[:,j].argsort()
        pointsE.append((min_id[0], min_id[1]))
    
    pointsE = list(set(pointsE))
    
    plt.figure()
    plt.subplot(121)
    plt.plot(*zip(*points),'o')
    plt.title("Phase Space: " + str(data_labels))
    plt.xlabel("Angle (Normalized)")
    plt.ylabel("Angular Velocity (Normalized)")
    plt.grid()
    plt.subplot(122)
    for edge in pointsE:
        a = pointsL[edge[0]]
        b = pointsL[edge[1]]
        plt.plot([a[0],b[0]], [a[1],b[1]], 'k', linewidth=3)
    plt.plot(*zip(*pointsL),'o', markersize=12)
    plt.title("Witness Complex: " + str(data_labels))
    plt.xlabel("Angle (Normalized)")
    plt.ylabel("Angular Velocity (Normalized)")
    plt.grid()