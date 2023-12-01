#!/usr/bin/env python3
# Produce the inputs tensors used in the signal selection NN.
# This file is part of https://github.com/hh-italian-group/hh-bbtautau.

import numpy as np
import ROOT
import os
import json
initialized = False
max_jet = 10

class SampleType:
    Data = 1
    MC = 2
    DY = 3
    QCD = 4
    TT = 5
    ggHH_NonRes = 6
    VBFHH_NonRes = 7
    ggHH_Res = 8
    VBFHH_Res = 9

def FindFiles(path, pattern) :
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if re.match(pattern, file):
                files.append(os.path.join(r, file))
                print(os.path.join(r, file))
    print(path)
    v = ROOT.std.vector('string')()
    for file in files:
        v.push_back(file)
    return v

def DefineVariables(sample_name, parity, use_deepTau_ordering) :
    global initialized
    if not initialized:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(file_dir)
        pyInterface_path = os.path.join(base_dir, 'include', 'pyInterface.h')
        ROOT.gInterpreter.Declare(f'#include "{pyInterface_path}"')
        initialized = True

    df = ROOT.ROOT.RDataFrame('Event', sample_name)

    if parity >= 0 and parity <= 1:
        df = df.Filter(f'event % 2 == {parity}')

    df = df.Define('spin', '{if (sample == 1 || sample == 3) return 0; else if (sample == 2 || sample == 4) return 2; else return -1; }') \
           .Define('mass_point', 'X_mass') \
           .Define('sample_type', 'ToLegacySampleType(sample)') \
           .Define('node', 'node_index') \
           .Define('channelId', 'ToLegacyChannel(channel)')
    df = df.Define('sample_year', '2018')
    df = df.Define('n_jets', 'RecoJet_pt.size()') \
           .Define('htt_scalar_pt', 'HttCandidate_leg0_pt + HttCandidate_leg1_pt') \
           .Define('htt_p4', 'getHTTp4(HttCandidate_leg0_pt, HttCandidate_leg0_eta, HttCandidate_leg0_phi, HttCandidate_leg0_mass,HttCandidate_leg1_pt, HttCandidate_leg1_eta, HttCandidate_leg1_phi, HttCandidate_leg1_mass)') \
           .Define('htt_pt', 'htt_p4.pt()') \
           .Define('htt_eta', 'htt_p4.eta()') \
           .Define('rel_met_pt_htt_pt', 'MET_pt / htt_scalar_pt') \
           .Define('htt_met_dphi', 'TVector2::Phi_mpi_pi(MET_phi - htt_p4.phi())') \
           # .Define('jets_genbJet', 'MakeGenbJet(jets_genJetIndex, jets_deepFlavourOrderedIndex)') \
           # .Filter('std::accumulate(jets_genbJet.begin(), jets_genbJet.end(), 0) == 2')

    for jet_idx in range(max_jet):
        df = df.Define(f'jet_{jet_idx}_valid', f'static_cast<float>({jet_idx}) < RecoJet_pt.size()') \
               .Define(f'jet_{jet_idx}_p4', f'jet_{jet_idx}_valid ? LorentzVectorM(RecoJet_pt.at({jet_idx}), RecoJet_eta.at({jet_idx}), RecoJet_phi.at({jet_idx}), RecoJet_mass.at({jet_idx})) : LorentzVectorM()') \
               .Define(f'jet_{jet_idx}_pt', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.pt() : 0.f') \
               .Define(f'jet_{jet_idx}_eta', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.eta() : 0.f') \
               .Define(f'jet_{jet_idx}_E', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.E() : 0.f') \
               .Define(f'jet_{jet_idx}_M', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.M() : 0.f') \
               .Define(f'rel_jet_{jet_idx}_M_pt', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.M() / jet_{jet_idx}_p4.pt() : 0.f') \
               .Define(f'rel_jet_{jet_idx}_E_pt', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.E() / jet_{jet_idx}_p4.pt() : 0.f') \
               .Define(f'jet_{jet_idx}_genbJet', f'jet_{jet_idx}_valid ? RecoJet_genMatched.at({jet_idx}) : 0.f') \
               .Define(f'jet_{jet_idx}_deepFlavour', f'jet_{jet_idx}_valid ? RecoJet_btagDeepFlavB.at({jet_idx}) : 0.f') \
               .Define(f'jet_{jet_idx}_ParticleNet', f'jet_{jet_idx}_valid ? RecoJet_particleNetAK4_B.at({jet_idx}) : 0.f') \
               .Define(f'jet_{jet_idx}_htt_dphi', f'jet_{jet_idx}_valid ? ROOT::Math::VectorUtil::DeltaPhi(htt_p4, jet_{jet_idx}_p4) : 0.f') \
               .Define(f'jet_{jet_idx}_htt_deta', f'jet_{jet_idx}_valid ? (htt_p4.eta()-jet_{jet_idx}_p4.eta()) : 0.f')
                # .Define(f'jet_{jet_idx}_deepCSV'.format(n_jet), 'jets_deepCSV(jets_deepCsv_BvsAll, jets_deepFlavourOrderedIndex, {})'.format(n_jet)) \

    return df

def CreateColums() :
    evt_columns = [ 'sample_type', 'spin', 'mass_point', 'node', 'sample_year', 'channelId', 'htt_pt', 'htt_eta',
                    'n_jets', 'htt_scalar_pt', 'htt_met_dphi', 'rel_met_pt_htt_pt'
    ]

    jet_column = [ 'jet_{}_valid', 'jet_{}_pt', 'jet_{}_eta', 'jet_{}_E', 'jet_{}_M', 'rel_jet_{}_M_pt', 'rel_jet_{}_E_pt',
                   'jet_{}_htt_deta', 'jet_{}_deepFlavour', 'jet_{}_ParticleNet', 'jet_{}_htt_dphi', 'jet_{}_genbJet',
    ]

    all_vars = evt_columns + jet_column
    jet_columns = []

    for jet_var in jet_column :
        for n in range(10) :
            jet_columns.append(jet_var.format(n))

    return evt_columns, jet_column, all_vars, jet_columns


def GetIndex(x) :
    evt_columns, jet_column, all_vars, jet_columns = CreateColums()
    if type(x) == list :
        all_indexes = []
        for var in range(len(x)) :
            var_name = x[var]
            idx = all_vars.index(var_name)
            all_indexes.append(idx)
        return all_indexes
    elif type(x) == str :
        idx = all_vars.index(x)
        return idx

def CreateInputs(raw_data):
    evt_columns, jet_column, all_vars, jet_columns = CreateColums()
    n_vars_evt = len(evt_columns)
    n_vars_jet = len(jet_column)
    n_evt = len(raw_data['n_jets'])

    data = np.zeros((n_evt, max_jet, n_vars_evt+n_vars_jet ), dtype=np.float32)

    evt_vars_idx = GetIndex(evt_columns)
    jet_vars_idx = GetIndex(jet_column)

    for jet_idx in range(max_jet):
        for n in range(len(evt_vars_idx)):
            data[:, jet_idx, evt_vars_idx[n]] = raw_data[all_vars[evt_vars_idx[n]]][:]
        for n in range(len(jet_vars_idx)):
            data[:, jet_idx, jet_vars_idx[n]] = raw_data[all_vars[jet_vars_idx[n]].format(jet_idx)][:]
    return data

def CreateRootDF(sample_name, parity, do_shuffle, use_deepTau_ordering):
    df = DefineVariables(sample_name, parity, use_deepTau_ordering)
    evt_columns, jet_column, all_vars, jet_columns = CreateColums()
    data_raw = df.AsNumpy(columns=evt_columns+jet_columns)
    data = CreateInputs(data_raw)
    if do_shuffle:
        np.random.shuffle(data)

    return data

def CreateXY(data, training_variables):
    with open(training_variables) as json_file:
        var = json.load(json_file)

    training_evt_vars     = var['evt_vars']
    idx_training_evt_vars =  GetIndex(training_evt_vars)

    training_jet_vars     =  var['jet_vars']
    idx_training_jet_vars =  GetIndex(training_jet_vars)

    training_vars         =   training_jet_vars + training_evt_vars
    training_vars_idx     =   idx_training_jet_vars + idx_training_evt_vars

    genTruth_var          = [ 'jet_{}_genbJet' ]
    idx_genTruth_var      =  GetIndex(genTruth_var)

    id_vars               = ['sample_type', 'spin', 'mass_point', 'node', 'sample_year', 'channelId']
    idx_id_vars           =  GetIndex(id_vars)

    X = data[:, :, training_vars_idx]
    Y = data[:, :, idx_genTruth_var]
    Z = data[:, :, idx_id_vars]

    var_pos = {}
    for n in range(len(training_vars)):
        var_pos[training_vars[n]] = n

    var_name = {}
    for n in range(len(training_vars)):
        var_name[n] = training_vars[n]


    var_pos_z = {}
    for n in range(len(id_vars)):
        var_pos_z[id_vars[n]] = n


    valid_pos = var_pos['jet_{}_valid']
    for jet_idx in range(X.shape[1]):
        for var_idx in range(X.shape[2]):
            X[:, jet_idx, var_idx] = X[:, jet_idx, var_idx] * X[:, jet_idx, valid_pos]

    return X, Y, Z, var_pos, var_pos_z, var_name
