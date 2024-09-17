import ROOT
import os
from ROOT import RDataFrame, TVector2

input_path = "/eos/user/e/emartinv/HHBtag_Training/training_skims_Run3/"
input_file = "2023BPix_GluGlutoHHto2B2Tau_SM.root"
input = os.path.join(input_path, input_file)    

output_dir = "output_variables"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ROOT.gInterpreter.Declare(f'#include "../include/pyInterface.h"')

df = RDataFrame("Event", input)

df = df.Define('n_jets', 'RecoJet_pt.size()') \
       .Define('htt_scalar_pt', 'HttCandidate_leg0_pt + HttCandidate_leg1_pt') \
       .Define('htt_p4', 'getHTTp4(HttCandidate_leg0_pt, HttCandidate_leg0_eta, HttCandidate_leg0_phi, HttCandidate_leg0_mass, HttCandidate_leg1_pt, HttCandidate_leg1_eta, HttCandidate_leg1_phi, HttCandidate_leg1_mass)') \
       .Define('htt_pt', 'htt_p4.Pt()') \
       .Define('htt_eta', 'htt_p4.Eta()') \
       .Define('rel_met_pt_htt_pt', 'MET_pt / htt_scalar_pt') \
       .Define('htt_met_dphi', 'TVector2::Phi_mpi_pi(MET_phi - htt_p4.Phi())') 

taggers = ["btagDeepFlavB", "btagPNetB", "btagRobustParTAK4B", "HHBtagScore"]
for tag in taggers:
    df = df.Define(f'{tag}_leading_jet_idx', f'ArgMax(RecoJet_{tag})') \
           .Define(f'{tag}_subleading_jet_idx', f'ArgMax(RecoJet_{tag}[RecoJet_{tag} != RecoJet_{tag}[{tag}_leading_jet_idx]])') \
           .Define(f'{tag}_lead_jet_valid', f'{tag}_leading_jet_idx < RecoJet_pt.size()') \
           .Define(f'{tag}_lead_jet_p4', f'LorentzVectorM(RecoJet_pt[{tag}_leading_jet_idx], RecoJet_eta[{tag}_leading_jet_idx], RecoJet_phi[{tag}_leading_jet_idx], RecoJet_mass[{tag}_leading_jet_idx])') \
           .Define(f'{tag}_lead_jet_pt', f'RecoJet_pt[{tag}_leading_jet_idx]') \
           .Define(f'{tag}_lead_jet_eta', f'RecoJet_eta[{tag}_leading_jet_idx]') \
           .Define(f'{tag}_lead_jet_phi', f'RecoJet_phi[{tag}_leading_jet_idx]') \
           .Define(f'{tag}_lead_jet_M', f'RecoJet_mass[{tag}_leading_jet_idx]') \
           .Define(f'{tag}_lead_jet_E', f'{tag}_lead_jet_p4.E()') \
           .Define(f'{tag}_lead_jet_rel_jet_M_pt', f'{tag}_lead_jet_p4.M() / {tag}_lead_jet_p4.Pt()') \
           .Define(f'{tag}_lead_jet_rel_jet_E_pt', f'{tag}_lead_jet_p4.E() / {tag}_lead_jet_p4.Pt()') \
           .Define(f'{tag}_lead_jet_genbJet', f'RecoJet_genMatched[{tag}_leading_jet_idx]') \
           .Define(f'{tag}_lead_jet_htt_dphi', f'TVector2::Phi_mpi_pi(htt_p4.Phi() - {tag}_lead_jet_p4.Phi())') \
           .Define(f'{tag}_lead_jet_htt_deta', f'htt_p4.Eta() - {tag}_lead_jet_p4.Eta()') \
           .Define(f'{tag}_lead_jet_{tag}', f'RecoJet_{tag}[{tag}_leading_jet_idx]') \
           .Define(f'{tag}_sublead_jet_{tag}', f'RecoJet_{tag}[{tag}_subleading_jet_idx]')

variables = ["htt_pt", "htt_eta", "htt_met_dphi", "rel_met_pt_htt_pt", "htt_scalar_pt"]
for tag in taggers:
    variables.extend([f"{tag}_lead_jet_valid", f"{tag}_lead_jet_pt", f"{tag}_lead_jet_eta", f"{tag}_lead_jet_phi", f"{tag}_lead_jet_M",
                      f"{tag}_lead_jet_E", f"{tag}_lead_jet_rel_jet_M_pt", f"{tag}_lead_jet_rel_jet_E_pt", f"{tag}_lead_jet_genbJet",
                      f"{tag}_lead_jet_htt_dphi", f"{tag}_lead_jet_htt_deta", f"{tag}_lead_jet_{tag}", f"{tag}_sublead_jet_{tag}"])

for var in variables:
    canvas = ROOT.TCanvas(var, var, 800, 600)
    hist = df.Histo1D(var)
    hist.Draw()
    
    first_bin = hist.FindFirstBinAbove(0)
    last_bin = hist.FindLastBinAbove(0)
    
    x_min = hist.GetXaxis().GetBinLowEdge(first_bin)
    x_max = hist.GetXaxis().GetBinUpEdge(last_bin)
    
    margin = (x_max - x_min) * 0.05
    x_min -= margin
    x_max += margin
    
    hist.GetXaxis().SetRangeUser(x_min, x_max)
    
    # canvas.SaveAs(f"{output_dir}/{var}_{input_file}.png")

output_file = ROOT.TFile(f"{output_dir}/histograms_{input_file}", "RECREATE")
for var in variables:
    hist = df.Histo1D(var)
    hist.Write()


tree = df.Snapshot("tree", f"{output_dir}/tree_{input_file}.root", [f'{tag}_lead_jet_{tag}' for tag in taggers] + [f'{tag}_sublead_jet_{tag}' for tag in taggers])

output_file.Close()
