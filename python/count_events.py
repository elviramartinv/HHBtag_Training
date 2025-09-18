import os
import ROOT

def count_events_in_root_file(file_path, tree_name='Event'):
    try:
        file = ROOT.TFile.Open(file_path)
        if file and not file.IsZombie():
            tree = file.Get(tree_name)
            if tree:
                return tree.GetEntries()
        return 0
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def count_events_in_folder(folder_path, tree_name='Event'):
    total_events = 0
    root_files = [f for f in os.listdir(folder_path) if f.endswith('.root')]
    for root_file in root_files:
        full_path = os.path.join(folder_path, root_file)
        n = count_events_in_root_file(full_path, tree_name)
        print(f"{root_file}: {n} events")
        total_events += n
    print(f"\nTotal events: {total_events}")
    return total_events

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python count_events.py /path/to/root/files [tree_name]")
        sys.exit(1)
    folder = sys.argv[1]
    tree = sys.argv[2] if len(sys.argv) > 2 else 'Event'
    count_events_in_folder(folder, tree)
