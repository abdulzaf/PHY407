"""
README
PHY 407 - Gait Distinction via Hip-Knee Coordination
Authors:
        Abdullah Zafar, 999730411
"""
The following is a project directory:

==========================================
Data and Video Files:
==========================================
	OriginalRuns.mp4 --> 5 run videos (1. sprinter, 2. sprinter, 3. paralympic sprinter, 4. marathon runner, 5. marathon runner) compiled from: https://www.youtube.com/user/LocomotorLabSMU
	data_sprint_1.txt --> data for run #1 (sprinter): hip angle | hip angular velocity | knee angle | knee angular velocity
	data_sprint_2.txt --> data for run #1 (sprinter): hip angle | hip angular velocity | knee angle | knee angular velocity
	data_sprint_para_1.txt --> data for run #1 (paralympic sprinter): hip angle | hip angular velocity | knee angle | knee angular velocity
	data_mar_1.txt --> data for run #4 (marathon runner): hip angle | hip angular velocity | knee angle | knee angular velocity
	data_mar_1.txt --> data for run #5 (marathon runner): hip angle | hip angular velocity | knee angle | knee angular velocity

==========================================
Program Files:
==========================================
	PHY407_Zafar_MainProgram.py --> Main program to run analysis
	PHY407_Zafar_Functions_Main.py --> File containing 3 main analysis functions: continuousrelphase, persistenthomology, wassersteindist
	PHY407_Zafar_Functions_Helper.py --> File containing general helper functions: norm, normalize, gennormpoints
	PHY407_Zafar_Functions_Plot.py --> File containing plotting functions: pltcrp, plotcrpphase, plotpersistencediagram, plotmatrix, plotwitcomplex
	PHY407_Zafar_Functions_CRP.py --> File containing helper functions for CRP analysis: processphasespace, relphasediff, 
	PHY407_Zafar_Functions_Topology.py --> File containing helper functions for persistent homology computation: witnesscomplex, vrfilt, createboundarymat, getpivotindices, reduceboundarymat, getintervals
	PHY407_Zafar_Functions_Wasserstein.py --> File containing helper functions for Wasserstein distance computation: projectdiagonals, exchangediagonals, createbipartitematrix, hungarianalgorithm, assignpairs, wassersteindistpairwise
==========================================
Test Case Files:
==========================================
	PHY407_Zafar_TestCases.py --> Set of test cases for witness complex generation, boundary matrix reduction and Hungarian algorithm implementation