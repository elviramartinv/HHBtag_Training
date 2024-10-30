#!/usr/bin/env python3
import numpy as np
import ROOT
import argparse
import os
from InputsProducer import CreateRootDF, CreateColums  

def save_to_root(data, output_file):
    evt_columns, jet_column, _, _ = CreateColums()
    
    root_file = ROOT.TFile(output_file, 'RECREATE')
    
    main_tree = ROOT.TTree("Event", "")

    branches_main = {}

    max_jet = data.shape[1]
    n_vars_evt = len(evt_columns)
    n_vars_jet = len(jet_column)

    for i, evt_var in enumerate(evt_columns):
        branches_main[evt_var] = np.zeros(1, dtype=np.float32)
        main_tree.Branch(evt_var, branches_main[evt_var], f"{evt_var}/F")

    for jet_idx in range(max_jet):
        for i, jet_var in enumerate(jet_column):
            branch_name = f'jet_{jet_idx}_{jet_var.format(jet_idx)}'
            branches_main[branch_name] = np.zeros(1, dtype=np.float32)
            main_tree.Branch(branch_name, branches_main[branch_name], f"{branch_name}/F")

    true_jets_tree = ROOT.TTree("TrueJetsTree", "Jets with genbJet == 1")

    branches_true_jets = {
        "jet_index": np.zeros(1, dtype=np.int32)  
    }
    true_jets_tree.Branch("jet_index", branches_true_jets["jet_index"], "jet_index/I")

    for jet_var in jet_column:
        branch_name = f"jet_{jet_var}"
        branches_true_jets[branch_name] = np.zeros(1, dtype=np.float32)
        true_jets_tree.Branch(branch_name, branches_true_jets[branch_name], f"{branch_name}/F")

    for event in data:
        for i, evt_var in enumerate(evt_columns):
            branches_main[evt_var][0] = event[0, i]

        for jet_idx in range(max_jet):
            for i, jet_var in enumerate(jet_column):
                branch_name = f'jet_{jet_idx}_{jet_var.format(jet_idx)}'
                branches_main[branch_name][0] = event[jet_idx, i + n_vars_evt]

        main_tree.Fill()

        for jet_idx in range(max_jet):
            genbJet_value = event[jet_idx, n_vars_evt + jet_column.index("jet_{}_genbJet")]
            if genbJet_value == 1:
                branches_true_jets["jet_index"][0] = jet_idx
                for i, jet_var in enumerate(jet_column):
                    branch_name = f"jet_{jet_var}"
                    branches_true_jets[branch_name][0] = event[jet_idx, i + n_vars_evt]
                true_jets_tree.Fill()

    root_file.Write()
    root_file.Close()

def main():
    parser = argparse.ArgumentParser(description="Process input data from file(s) and save to ROOT file")
    parser.add_argument("--inFile", help="Path to a single input ROOT file")
    parser.add_argument("--inDir", help="Path to a directory with multiple input ROOT files")
    parser.add_argument("--outFile", required=True, help="Path to save the output ROOT file")
    parser.add_argument("--parity", type=int, default=0, help="Parity (0 or 1) to filter events")

    args = parser.parse_args()

    input_files = []
    if args.inFile:
        input_files.append(args.inFile)
    elif args.inDir:
        input_files = [os.path.join(args.inDir, f) for f in os.listdir(args.inDir) if f.endswith(".root")]
    else:
        raise ValueError("Specify --inFile o --inDir")

    all_data = []

    for input_file in input_files:
        data = CreateRootDF(input_file, args.parity, do_shuffle=False, use_deepTau_ordering=False)
        all_data.append(data)

    all_data = np.concatenate(all_data, axis=0)

    save_to_root(all_data, args.outFile)
    print(f"Saved in {args.outFile}")

if __name__ == "__main__":
    main()
