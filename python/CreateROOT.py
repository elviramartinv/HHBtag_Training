import numpy as np
import ROOT
import argparse
from InputsProducer import CreateRootDF, CreateColums 

def save_to_root(data, output_file):
    evt_columns, jet_column, _, _ = CreateColums()
    
    root_file = ROOT.TFile(output_file, 'RECREATE')
    tree = ROOT.TTree("Event", "")

    max_jet = data.shape[1]
    n_vars_evt = len(evt_columns)
    n_vars_jet = len(jet_column)

    branches = {}

    for i, evt_var in enumerate(evt_columns):
        branches[evt_var] = np.zeros(1, dtype=np.float32)
        tree.Branch(evt_var, branches[evt_var], f"{evt_var}/F")

    for jet_idx in range(max_jet):
        for i, jet_var in enumerate(jet_column):
            branch_name = f'jet_{jet_idx}_{jet_var.format(jet_idx)}'
            branches[branch_name] = np.zeros(1, dtype=np.float32)
            tree.Branch(branch_name, branches[branch_name], f"{branch_name}/F")

    for event in data:
        for i, evt_var in enumerate(evt_columns):
            branches[evt_var][0] = event[0, i]  

        for jet_idx in range(max_jet):
            for i, jet_var in enumerate(jet_column):
                branch_name = f'jet_{jet_idx}_{jet_var.format(jet_idx)}'
                branches[branch_name][0] = event[jet_idx, i + n_vars_evt]  

        tree.Fill()

    root_file.Write()
    root_file.Close()

def main():
    parser = argparse.ArgumentParser(description="Process input data and save to ROOT file")
    parser.add_argument("--inFile", required=True, help="Path to the input ROOT file")
    parser.add_argument("--outFile", required=True, help="Path to save the output ROOT file")
    parser.add_argument("--parity", type=int, default=0, help="Parity (0 or 1) to filter events")

    args = parser.parse_args()

    data = CreateRootDF(args.inputFile, args.parity, do_shuffle=False, use_deepTau_ordering=False)

    save_to_root(data, args.outputFile)
    print(f"Saved in {args.outputFile}")

if __name__ == "__main__":
    main()
