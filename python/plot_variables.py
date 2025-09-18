import ROOT
import os
import numpy as np
import matplotlib.pyplot as plt

input_folder = "output_root/"
tree_name = "Event"

root_files = [f for f in os.listdir(input_folder) if f.endswith(".root")]

# Construir lista de variables dinámicamente
jet_vars = ["pt", "eta", "E", "M", "htt_deta", "ParticleNet", "isbjet", "genvbfJet"]
rel_jet_vars = ["M_pt", "E_pt"]
isbjet_vars = ["isbjet_pt", "isbjet_eta", "isbjet_E", "isbjet_M"]
isvbfjet_vars = ["isvbfjet_pt", "isvbfjet_eta", "isvbfjet_E", "isvbfjet_M"]

variables = [f'jet_{i}_{var}' for i in range(10) for var in jet_vars] + \
            [f'rel_jet_{i}_{var}' for i in range(10) for var in rel_jet_vars] + \
            [f'jet_{i}_{var}' for i in range(10) for var in isbjet_vars] + \
            [f'jet_{i}_{var}' for i in range(10) for var in isvbfjet_vars]

data = {var: [] for var in variables}

for root_file in root_files:
    file_path = os.path.join(input_folder, root_file)
    rdf = ROOT.RDataFrame(tree_name, file_path)
    for var in variables:
        try:
            raw_values = np.array(rdf.AsNumpy(columns=[var])[var])  # Convertir en array NumPy
            filtered_values = raw_values[raw_values > 0]  # Filtrar valores no físicos (-999 y 0)
            data[var].append(filtered_values)
        except KeyError:
            print(f"Advertencia: La variable '{var}' no está en {root_file}")

# Graficar histogramas con normalización por número de eventos y sin relleno
for var in variables:
    plt.figure(figsize=(8, 6))
    
    for i, file_name in enumerate(root_files):
        if len(data[var]) > i and len(data[var][i]) > 0:
            num_events = len(data[var][i])  # Número total de eventos
            plt.hist(data[var][i], bins=20, weights=np.ones(num_events) / num_events,
                     histtype='step', linewidth=2, label=file_name)  # Modo "step" para solo contorno
    
    plt.xlabel(var)
    plt.ylabel("Normalized Events")
    plt.yscale('log')  # Escala logarítmica para el eje y
    plt.title(f"{var}")
    plt.legend()
    plt.grid(True)
    
    # Guardar la imagen
    plt.savefig(f"output_root/histograms/{var}.png")
    plt.close()

print("¡Histogramas normalizados generados con contorno y guardados en 'output_root/histograms/'!")
