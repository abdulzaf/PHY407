"""
MAIN PROGRAM
PHY 407 - Gait Distinction via Hip-Knee Coordination
Authors:
        Abdullah Zafar, 999730411
"""
from PHY407_Zafar_Functions_Main import *
from PHY407_Zafar_Functions_Plot import *
import numpy as np
import matplotlib.pyplot as plt

# Increase plot font size
plt.rcParams.update({'font.size': 12})
plt.close('all')

# Set filenames to load data for each trial from
filenames = [
    'data_sprint_1.txt',
    'data_sprint_2.txt',
    'data_sprint_para_1.txt',
    'data_mar_1.txt',
    'data_mar_2.txt',
    ]
# Set labels for each trial
data_labels = [
    'sprint_1',
    'sprint_2',
    'sprint_P',
    'mara_1',
    'mara_2',
    ]

# Initialize values and lists
intervals = [] # List to hold homology intervals for each trial
crp = [] # List to hold continuous relative phase (CRP) angle for each trial
crpdot = [] # List to hold CRP angular velocity for each trial
xrange = [] # List to hold range of CRP angle values across trials
yrange = [] # List to hold range of CRP angular velocity values across trials
# Witness complex parameters
# Vietoris-Rips filtration parameters
nlandmarks = 10 # Number of landmark points to use in generating the witness complex
start = 0 # Starting epsilon
end = 3 # Stopping epsilon
step = 0.01 # Step size for filtration
# Wasserstein computation parameters
k = 3 # Use the k largest intervals for comparison
A_thresh = 0.5 # Set adjacency threshold at 0.5 (50%)

# PARSE DATA FILES AND COMPUTE CONTINUOUS RELATIVE PHASE
for i in range(len(filenames)):
    # Load Data Files
    raw = np.loadtxt(filenames[i])
    
    # Compute continuous relative phase and store
    x, xdot = continuousrelphase(raw, data_labels[i])
    crp.append(x)
    crpdot.append(xdot)
    
    # Keep the data ranges for cross-trial normalization later
    xrange.append(min(x))
    xrange.append(max(x))
    yrange.append(min(xdot))
    yrange.append(max(xdot))

# Find the range of data over all trials for cross-trial normalization
xminmax = [min(xrange), max(xrange)]
yminmax = [min(yrange), max(yrange)]

# COMPUTE PERSISTENT HOMOLOGY FOR EACH TRIAL
for i in range(len(filenames)):
    # Compute persistent homology
    interval_trial = persistenthomology(crp[i], crpdot[i], xminmax, yminmax, nlandmarks, start, step, end, False, data_labels[i])
    # Store homology intervals for comparing later
    intervals.append(interval_trial)


# COMPUTE WASSERSTEIN DISTANCE BETWEEN EACH PAIR OF INTERVALS
# Compute the distance matrix, using the Wasserstein distance metric
W = wassersteindist(intervals, k)
# Bring matrix values to [0,1] using maximum
W /= np.max(W)

# Initialize adjacency matrix
A = np.zeros((len(W),len(W)))
# Compute adjacency matrix
A[W<A_thresh] = 1
np.fill_diagonal(A, 0)
    
#% Plots
plotcrp(crp, data_labels)
plotcrpphase(crp, crpdot, data_labels)
plotpersistencediagram(0.5,2,intervals,k,data_labels)
plotmatrix(W, data_labels, "Wasserstein Distance Matrix (Normalized)")
plotmatrix(A, data_labels, "Adjacency Matrix (threshold=0.5)")


