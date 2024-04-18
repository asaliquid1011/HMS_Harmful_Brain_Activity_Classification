# -*- coding: utf-8 -*-

import numpy as np

# ===============
# math
# ===============
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return int(np.log2(2**m_i))

def calcNorm(Data):
    Mean=np.mean(Data,axis=0)
    Std=np.std(Data,axis=0)
    Data_norm=(Data-Mean)/Std
    return Mean,Std,Data_norm
#
def calcCov(Data_norm):
    Mean_norm = np.mean(Data_norm, axis=0)
    cov_norm = np.cov(Data_norm.T)
    Inv_cov_norm=np.linalg.pinv(cov_norm)
    return Mean_norm ,Inv_cov_norm

