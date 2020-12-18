"""
TEST CASES
PHY 407 - Gait Distinction via Hip-Knee Coordination
Authors:
        Abdullah Zafar, 999730411
"""
from PHY407_Zafar_Functions_Helper import *
from PHY407_Zafar_Functions_Plot import *
from PHY407_Zafar_Functions_Topology import *
from PHY407_Zafar_Functions_Wasserstein import *

"""
Test Witness Complex Generation 
Input: 100 points on a circle, 4 landmarks
Expected Output: square witness complex
"""
t = np.linspace(0,2*np.pi,100)
x = np.cos(t)
y = np.sin(t)
points = gennormpoints(x,y,[min(x),max(x)],[min(y),max(y)])
pointsL = witnesscomplex(points, 4)

plotwitcomplex(points, pointsL, "")

"""
Test Boundary Matrix Reduction
Input:
    [[0 0 0 1 1 0 0],
     [0 0 0 1 0 1 0],
     [0 0 0 0 1 1 0],
     [0 0 0 0 0 0 1],
     [0 0 0 0 0 0 1],
     [0 0 0 0 0 0 1],
     [0 0 0 0 0 0 0]]
Expected Output:
    [[0 0 0 1 1 0 0],
     [0 0 0 1 0 0 0],
     [0 0 0 0 1 0 0],
     [0 0 0 0 0 0 1],
     [0 0 0 0 0 0 1],
     [0 0 0 0 0 0 1],
     [0 0 0 0 0 0 0]]
"""
delta = np.array(
    [[0, 0, 0, 1, 1, 0, 0],
     [0, 0, 0, 1, 0, 1, 0],
     [0, 0, 0, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0]])
delta_red = reduceboundarymat(delta)

print("Reduced Boundary Matrix: " + str(delta_red))


"""
Test Hungarian Algorithm Implementation
Input:
    [[1500 4000 4500],
     [2000 6000 3500],
     [2000 4000 2500]]
Expected Output:
    [[   0    0 2000],
     [   0 1500  500],
     [ 500    0    0]]
    
"""
C = np.array(
    [[1500, 4000, 4500],
     [2000, 6000, 3500],
     [2000, 4000, 2500]])
C_red = hungarianalgorithm(C)
print("Reduced Cost Matrix: " + str(C_red))