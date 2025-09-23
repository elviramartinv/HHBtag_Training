import uproot
import os


# folder = "/eos/user/e/emartinv/HHBtag_Training/training_skims_2024/"
folder = "/eos/user/e/emartinv/HHBtag_Training/training_skims_Run3_0109//"

tree_name = "Event"

for filename in os.listdir(folder):
    if filename.endswith(".root"):
        filepath = os.path.join(folder, filename)
        try:
            with uproot.open(filepath) as file:
                tree = file[tree_name]
                n_entries = tree.num_entries
                print(f"{filename}: {n_entries} entries")
        except Exception as e:
            print(f"Error in {filename}: {e}")
