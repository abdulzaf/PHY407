"""
Helper Functions - Continuous Relative Phase Computation
PHY 407 - Gait Distinction via Hip-Knee Coordination
Authors:
        Abdullah Zafar, 999730411
"""

from PHY407_Zafar_Functions_Helper import normalize
import numpy as np

def processphasespace(raw):
    """
    Given a formatted array of data, return the normalized data

    Parameters
    ----------
    raw : array[float], size: Nx4
        N - number of observations
        4 - series of data in the following order:
            column 1: Hip angle
            column 2: Hip angular velocity
            column 3: Knee angle
            column 4: Knee angular velocity

    Returns
    -------
    thHN : array[float], size: Nx1
        normalized hip angles
    omHN : array[float], size: Nx1
        normalized hip angular velocities
    thKN : array[float], size: Nx1
        normalized knee angles
    omKN : array[float], size: Nx1
        normalized knee angular velocities

    """
    N = len(raw)
    thH = raw[:,0]
    omH = raw[:,1]
    thK = raw[:,2]
    omK = raw[:,3]
    
    thHN = normalize(thH, [min(thH),max(thH)])
    omHN = normalize(omH, [min(omH),max(omH)])
    thKN = normalize(thK, [min(thK),max(thK)])
    omKN = normalize(omK, [min(omK),max(omK)])
    
    return thHN, omHN, thKN, omKN

def relphasediff(x,xdot,y,ydot):
    """
    Compute the relative phase between two phase spaces

    Parameters
    ----------
    x : array[float], size: Nx1
        first dimension of the first phase space
    xdot : array[float], size: Nx1
        second dimension of the first phase space
    y : array[float], size: Nx1
        first dimension of the second phase space
    ydot : array[float], size: Nx1
        second dimension of the second phase space

    Returns
    -------
    omega : array[float], size: Nx1
        array of the continuous relative phase of the two phase spaces

    """
    # Compute individual phase angles and then compute difference
    phasex = np.arctan2(xdot,x)
    phasey = np.arctan2(ydot,y)
    omega = phasex-phasey
    
    # Correct phase values
    omega = omega * 180 / np.pi
    omega[omega<-180] += 360
    
    return omega
