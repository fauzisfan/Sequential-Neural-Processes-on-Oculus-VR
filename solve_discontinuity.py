import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import skinematics as skin
import matplotlib.pyplot as plt
import math

#from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers

def solve_discontinuity(eul):
    eul = np.deg2rad(eul)   # convert to radian
    X_eul = eul[:,0]
    Y_eul = eul[:,1]
    Z_eul = eul[:,2]
    
#    print(X_eul)
#    print(len(X_eul))
    
    arcsinX = np.zeros(len(X_eul))
    arcsinY = np.zeros(len(Y_eul))
    arcsinZ = np.zeros(len(Z_eul))
    for i in range(0,len(X_eul)):
    # take arcsin of it
    # X = pitch [-180,180]
        if (0 <= np.rad2deg(X_eul[i]) < 90): # Quadrant I
            arcsinX[i] = np.arcsin(np.sin(X_eul[i]))
        elif (90 <= np.rad2deg(X_eul[i]) < 180): #quadrant II
            arcsinX[i] = np.deg2rad(180) - np.arcsin(np.sin(X_eul[i]))
        elif (-180 <= np.rad2deg(X_eul[i]) < -90): #quadrant III
            arcsinX[i] = -np.deg2rad(180) - np.arcsin(np.sin(X_eul[i]))
        else: # quadrant IV
            arcsinX[i] = np.arcsin(np.sin(X_eul[i]))
    
    # Y = roll [-90,90] leave it alone
    for i in range(0,len(Y_eul)):
        arcsinY[i] = np.arcsin(np.sin(Y_eul[i]))
    
    # Z = yaw [-180,180] => see slide
    for i in range(0,len(Z_eul)):
        if (0 <= np.rad2deg(Z_eul[i]) < 90): # Quadrant I 
            arcsinZ[i] = np.arcsin(np.sin(Z_eul[i]))
        elif (90 <= np.rad2deg(Z_eul[i]) < 180): #Quadran II
            arcsinZ[i] = np.deg2rad(180) - np.arcsin(np.sin(Z_eul[i]))
        elif (-180 <= np.rad2deg(Z_eul[i]) < -90): #quadrant III
            arcsinZ[i] = -np.deg2rad(180) - np.arcsin(np.sin(Z_eul[i]))
        else: # Quadrant IV
            arcsinZ[i] = np.arcsin(np.sin(Z_eul[i]))
    
    # get the euler
    eul = np.rad2deg(np.column_stack((arcsinX,arcsinY,arcsinZ)))
    return eul