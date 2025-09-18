import os
import pandas as pd
import numpy as np
import numba
from scipy import interpolate

@numba.njit
def calculeteNEvents(X, Z):
    mass_node_unique = np.unique(Z[:, 0, 1])
    counts = np.zeros((4,6,len(mass_node_unique)))
    print("Z", Z.shape, "and values", Z)
    for n in range(X.shape[0]):
        year = int(Z[n, 0, -2])#var_pos['era_id']]
        channel = int(Z[n, 0, -1])#var_pos['pairType']]
        mn = Z[n, 0, 1]
        mn_index = np.argmax(mass_node_unique == mn)
        # channel_idx = channel
        counts[year, channel, mn_index] +=1
    weights = np.where(counts > 0, 1. / counts, 0)

    return counts, weights, mass_node_unique

def CreateXSTable(X, Z, version="equal"):

    counts, weights, mass_node_unique = calculeteNEvents(X, Z)

    xs_values = {
        0: 0.0001415778, #SM 1-1-1
        1: 0.0021778912, # 1-0-1
        2: 0.0293703335, # 1.74-1.37-14.4
        3: 0.0000009337, # -0.012-0.030-10.2
        4: 0.0263694193, # -0.758-1.44--19.3
        5: 0.0000827480, # -0.962-0.959--1.43
        6: 0.0002787730, # -1.21-1.94--0.94
        7: 0.0008586774, # -1.60-2.72--1.36
        8: 0.0012367629, # -1.83-3.57--3.39
        9: 0.0499087685, # -2.12-3.87--5.96
    }

    xs_all = np.ones((4,6,len(mass_node_unique)))

    if version == "inv_xsec":
        for i, node in enumerate(mass_node_unique):
            if node in xs_values:
                xs_all[:, :, i] = 1.0 / xs_values[node]
            else:
                print(f"Warning: Not XSec assigned for node {node}!")
    
    elif version == "equal":
        for i in range(len(mass_node_unique)):
            xs_all[:, :, i] = 1.0

    elif version == "sm_boost":
        xs_all[:, :, mass_node_unique == 0] *= 100  # SM Boost

    return xs_all, counts, weights, mass_node_unique

def CreateWeights(X, Z):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(file_dir)
    xs_all, counts, weights, mass_node_unique = CreateXSTable(X, Z, version="sm_boost")

    for node in range(len(mass_node_unique)):
        weights[:, :, node] *= xs_all[:,:,node]
    print("CreateWeights: weights after xs_all assignment", weights[:,:,0])


    for year in range(4):
        ch_cnt = np.zeros(6)  
        ref_ch = 0 

        main_channels = [0, 1, 2]  # eTau, muTau, tauTau
        low_stat_channels = [3, 4, 5]  # ee, emu, mumu

        for channel_id in main_channels + low_stat_channels:
            ch_cnt[channel_id] = np.sum(weights[year, channel_id, :])
            print(f'CreateWeights: year={year} channel_id={channel_id} ch_cnt={ch_cnt[channel_id]}')

        ref_ch = np.amax([ch_cnt[channel] for channel in main_channels])
        print(f'CreateWeights: year={year} ref_ch={ref_ch} ch_cnt={ch_cnt}')
        if ref_ch == 0: 
            print(f"Warning: All main channels have 0 events in year {year}")
            continue

        print(f'CreateWeights: ch_cnt = {ch_cnt}')

        for channel_id in main_channels:
            if ch_cnt[channel_id] != 0:
                A = ref_ch / ch_cnt[channel_id]
                weights[year, channel_id, :] *= A

        for channel_id in low_stat_channels:
            if ch_cnt[channel_id] != 0:
                A = (ref_ch / ch_cnt[channel_id]) 
                weights[year, channel_id, :] *= A

    # all years
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/TWikiLUM
    luminosity = [7.980, 23.589, 18.063, 9.693]
    year_cnt = np.zeros(4)
    for year in range(4):
         year_cnt[year] = np.sum(weights[year, :, :] * counts[year, :, :])

    for year in range(4):
        if year_cnt[year] != 0 :
            A = luminosity[year] / year_cnt[year]
            print(f'CreateWeights: all years year={year} factor={A}')
            weights[year, :, :] *= A

    #sum all = n_evt
    weights_count = np.sum(weights * counts)
    C =  X.shape[0] / weights_count
    weights *= C


    return weights, mass_node_unique

@numba.njit
def ConvertToVector(X,Z, weights, mass_node_unique):
    weight_vec = np.zeros(X.shape[0])
    for n in range(X.shape[0]):
        year = int(Z[n, 0, -2])
        channel = int(Z[n, 0, -1])
        mn = int(Z[n, 0, 1])
        mn_index = np.argmax(mass_node_unique == mn)
        weight_vec[n] = weights[year, channel, mn_index]
    return weight_vec

def CreateSampleWeigts(X,Z):
    weights, mass_node_unique = CreateWeights(X, Z)
    return ConvertToVector(X,Z, weights, mass_node_unique)

@numba.njit
def CrossCheckWeights(Z, X, w, check_channel, year):
    w_1 = []
    w_2 = []
    w_3 = []
    w_4 = []
    w_5 = []
    w_6 = []

    if check_channel == True:
        for n in range(X.shape[0]):
            if Z[n, 0, -1] == 0 and Z[n, 0, -2] == year:
                w_1.append(w[n])
            elif Z[n, 0, -1] == 1 and Z[n, 0, -2] == year:
                w_2.append(w[n])
            elif Z[n, 0, -1] == 2 and Z[n, 0, -2] == year:
                w_3.append(w[n])
            elif Z[n, 0, -1] == 3 and Z[n, 0, -2] == year:
                w_4.append(w[n])
            elif Z[n, 0, -1] == 4 and Z[n, 0, -2] == year:
                w_5.append(w[n])
            elif Z[n, 0, -1] == 5 and Z[n, 0, -2] == year:
                w_6.append(w[n])

    return np.sum(np.array(w_1)), np.sum(np.array(w_2)), np.sum(np.array(w_3)), np.sum(np.array(w_4)), np.sum(np.array(w_5)), np.sum(np.array(w_6))
