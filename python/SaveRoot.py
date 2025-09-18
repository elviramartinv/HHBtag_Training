#!/usr/bin/env python3
# Produce the inputs tensors used in the signal selection NN.
# This file is part of https://github.com/hh-italian-group/hh-bbtautau.

import numpy as np
import ROOT
import os
import json
initialized = False
max_jet = 10

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

def DefineVariables(sample_name, parity) :
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
           .Define('pairType', 'channel') 
    df = df.Define('era_id', 'ToEraId(period)')      
    df = df.Define('n_jets', 'RecoJet_pt.size()') \
           .Define('htt_scalar_pt', 'HttCandidate_leg0_pt + HttCandidate_leg1_pt') \
           .Define('htt_p4', 'getHTTp4(HttCandidate_leg0_pt, HttCandidate_leg0_eta, HttCandidate_leg0_phi, HttCandidate_leg0_mass,HttCandidate_leg1_pt, HttCandidate_leg1_eta, HttCandidate_leg1_phi, HttCandidate_leg1_mass)') \
           .Define('htt_pt', 'htt_p4.pt()') \
           .Define('htt_eta', 'htt_p4.eta()') \
           .Define('rel_met_pt_htt_pt', 'PFMET_pt / htt_scalar_pt') \
           .Define('htt_met_dphi', 'TVector2::Phi_mpi_pi(PFMET_phi - htt_p4.phi())') \
           # .Define('jets_genbJet', 'MakeGenbJet(jets_genJetIndex, jets_deepFlavourOrderedIndex)') \
           # .Filter('std::accumulate(jets_genbJet.begin(), jets_genbJet.end(), 0) == 2')

    if args.nobjets:
        for jet_idx in range(max_jet):
            df = df.Define(f'jet_{jet_idx}_valid', f'static_cast<float>({jet_idx}) < RecoJet_pt.size() && !RecoJet_isCCLUBbjet.at({jet_idx})') \
                .Define(f'jet_{jet_idx}_p4', f'jet_{jet_idx}_valid ? LorentzVectorM(RecoJet_pt.at({jet_idx}), RecoJet_eta.at({jet_idx}), RecoJet_phi.at({jet_idx}), RecoJet_mass.at({jet_idx})) : LorentzVectorM()') \
                .Define(f'jet_{jet_idx}_pt', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.pt() : 0.f') \
                .Define(f'jet_{jet_idx}_eta', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.eta() : 0.f') \
                .Define(f'jet_{jet_idx}_E', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.E() : 0.f') \
                .Define(f'jet_{jet_idx}_M', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.M() : 0.f') \
                .Define(f'rel_jet_{jet_idx}_M_pt', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.M() / jet_{jet_idx}_p4.pt() : 0.f') \
                .Define(f'rel_jet_{jet_idx}_E_pt', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.E() / jet_{jet_idx}_p4.pt() : 0.f') \
                .Define(f'jet_{jet_idx}_genvbfJet', f'jet_{jet_idx}_valid ? RecoJet_vbfgenMatched.at({jet_idx}) : 0.f') \
                .Define(f'jet_{jet_idx}_deepFlavour', f'jet_{jet_idx}_valid ? RecoJet_btagDeepFlavB.at({jet_idx}) : 0.f') \
                .Define(f'jet_{jet_idx}_ParticleNet', f'jet_{jet_idx}_valid ? RecoJet_btagPNetB.at({jet_idx}) : 0.f') \
                .Define(f'jet_{jet_idx}_htt_dphi', f'jet_{jet_idx}_valid ? ROOT::Math::VectorUtil::DeltaPhi(htt_p4, jet_{jet_idx}_p4) : 0.f') \
                .Define(f'jet_{jet_idx}_htt_deta', f'jet_{jet_idx}_valid ? (htt_p4.eta()-jet_{jet_idx}_p4.eta()) : 0.f') \
                .Define(f'jet_{jet_idx}_isbjet', f'jet_{jet_idx}_valid ? RecoJet_isCCLUBbjet.at({jet_idx}) : 0.f') \
                .Define(f'jet_{jet_idx}_isbjet_pt', f'jet_{jet_idx}_isbjet ? jet_{jet_idx}_pt : -999.f') \
                .Define(f'jet_{jet_idx}_isbjet_eta', f'jet_{jet_idx}_isbjet ? jet_{jet_idx}_eta : -999.f') \
                .Define(f'jet_{jet_idx}_isbjet_E', f'jet_{jet_idx}_isbjet ? jet_{jet_idx}_E : -999.f') \
                .Define(f'jet_{jet_idx}_isbjet_M', f'jet_{jet_idx}_isbjet ? jet_{jet_idx}_M : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_pt', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_pt : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_eta', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_eta : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_E', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_E : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_M', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_M : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_PNet', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_ParticleNet : -999.f') \

    else:
        for jet_idx in range(max_jet):
            df = df.Define(f'jet_{jet_idx}_valid', f'static_cast<float>({jet_idx}) < RecoJet_pt.size()') \
                .Define(f'jet_{jet_idx}_p4', f'jet_{jet_idx}_valid ? LorentzVectorM(RecoJet_pt.at({jet_idx}), RecoJet_eta.at({jet_idx}), RecoJet_phi.at({jet_idx}), RecoJet_mass.at({jet_idx})) : LorentzVectorM()') \
                .Define(f'jet_{jet_idx}_pt', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.pt() : 0.f') \
                .Define(f'jet_{jet_idx}_eta', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.eta() : 0.f') \
                .Define(f'jet_{jet_idx}_E', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.E() : 0.f') \
                .Define(f'jet_{jet_idx}_M', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.M() : 0.f') \
                .Define(f'rel_jet_{jet_idx}_M_pt', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.M() / jet_{jet_idx}_p4.pt() : 0.f') \
                .Define(f'rel_jet_{jet_idx}_E_pt', f'jet_{jet_idx}_valid ? jet_{jet_idx}_p4.E() / jet_{jet_idx}_p4.pt() : 0.f') \
                .Define(f'jet_{jet_idx}_genvbfJet', f'jet_{jet_idx}_valid ? RecoJet_vbfgenMatched.at({jet_idx}) : 0.f') \
                .Define(f'jet_{jet_idx}_deepFlavour', f'jet_{jet_idx}_valid ? RecoJet_btagDeepFlavB.at({jet_idx}) : 0.f') \
                .Define(f'jet_{jet_idx}_ParticleNet', f'jet_{jet_idx}_valid ? RecoJet_btagPNetB.at({jet_idx}) : 0.f') \
                .Define(f'jet_{jet_idx}_htt_dphi', f'jet_{jet_idx}_valid ? ROOT::Math::VectorUtil::DeltaPhi(htt_p4, jet_{jet_idx}_p4) : 0.f') \
                .Define(f'jet_{jet_idx}_htt_deta', f'jet_{jet_idx}_valid ? (htt_p4.eta()-jet_{jet_idx}_p4.eta()) : 0.f') \
                .Define(f'jet_{jet_idx}_isbjet', f'jet_{jet_idx}_valid ? RecoJet_isCCLUBbjet.at({jet_idx}) : 0.f') \
                .Define(f'jet_{jet_idx}_isbjet_pt', f'jet_{jet_idx}_isbjet ? jet_{jet_idx}_pt : -999.f') \
                .Define(f'jet_{jet_idx}_isbjet_eta', f'jet_{jet_idx}_isbjet ? jet_{jet_idx}_eta : -999.f') \
                .Define(f'jet_{jet_idx}_isbjet_E', f'jet_{jet_idx}_isbjet ? jet_{jet_idx}_E : -999.f') \
                .Define(f'jet_{jet_idx}_isbjet_M', f'jet_{jet_idx}_isbjet ? jet_{jet_idx}_M : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_pt', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_pt : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_eta', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_eta : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_E', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_E : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_M', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_M : -999.f') \
                .Define(f'jet_{jet_idx}_isvbfjet_PNet', f'jet_{jet_idx}_genvbfJet == 1 ? jet_{jet_idx}_ParticleNet : -999.f') \

                # .Define(f'jet_{jet_idx}_deepCSV'.format(n_jet), 'jets_deepCSV(jets_deepCsv_BvsAll, jets_deepFlavourOrderedIndex, {})'.format(n_jet)) \

    return df

def SaveRoot(df, output_filename, tree_name="Event"):
    if args.nobjets:
        output_filename = output_filename.replace('.root', '_nobjets.root')
    else:
        output_filename = output_filename
        
    df.Snapshot(tree_name, output_filename)
    print(f"Archivo ROOT guardado en: {output_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Procesar ROOT y guardar el DataFrame modificado.')
    parser.add_argument('--input_file', '-i', type=str, help='Archivo ROOT de entrada')
    parser.add_argument('-p', '--parity', type=int, default=-1, choices=[-1,0,1], help='Filtrar evento por paridad (0 o 1), -1 sin filtro')
    parser.add_argument('-o', '--output', type=str, default='output_root/output.root', help='Archivo ROOT de salida')
    parser.add_argument('-nb','--nobjets', action='store_true', help='Remove selected bjets for input sample')
    args = parser.parse_args()

    df = DefineVariables(args.input_file, args.parity)
    SaveRoot(df, args.output)
