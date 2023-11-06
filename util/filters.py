import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfreqz, sosfilt




def MultiNotch_50Hz_60Hz(x):
    B1_60 =  0.95210183
    B2_60 = -0.11971715
    B3_60 =  0.95210183
    A2_60 = -0.11971714
    A3_60 =  0.90420359
    
    B1_50 =  0.95210183
    B2_50 = -0.58917546
    B3_50 =  0.95210183
    A2_50 = -0.58917540
    A3_50 =  0.90420359
    
    x1,y1,z1,x2,y2,z2,x3,y3,z3=[0]*9
    x_filtered= np.zeros(len(x))
    for f,x3 in enumerate(x):  
    
        y3 = B1_60 * x3 + B2_60 * x2 + B3_60 * x1 - A2_60 * y2 - A3_60 * y1
        z3 = B1_50 * y3 + B2_50 * y2 + B3_50 * y1 - A2_50 * z2 - A3_50 * z1
        
        x_filtered[f]=z3
        y1 = y2;
        y2 = y3;
        x1 = x2;
        x2 = x3;
        z1 = z2;
        z2 = z3; 
        
    return x_filtered


def BandPassFilter1_3_30Hz(x):
    
    B1_HI = 0.96384585
    B2_HI = -0.96384585
    A2_HI = -0.92769164
    
    B1_LOW = 0.09243869
    B2_LOW = 0.18487774
    B3_LOW = 0.09243869
    A2_LOW = -0.97528231
    A3_LOW = 0.34503776
    
    x1,y1,z1,x2,y2,z2,x3,y3,z3=[0]*9
    #x_filtered= np.zeros(x.shape[0])
    x_filtered= np.zeros(len(x))
    for f,x3 in enumerate(x):  
        
        y3 = B1_HI * x3 + B2_HI * x2 - A2_HI * y2
        z3 = B1_LOW * y3 + B2_LOW * y2 + B3_LOW * y1 - A2_LOW * z2 - A3_LOW * z1
        x_filtered[f]=z3
        y1 = y2;
        y2 = y3;
        x1 = x2;
        x2 = x3;
        z1 = z2;
        z2 = z3; 
        
    return x_filtered

    
    
    
    
    
        