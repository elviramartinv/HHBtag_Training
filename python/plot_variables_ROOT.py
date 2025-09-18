import ROOT

# Lista de archivos
files = [
    "output_root/2022_VBFHHto2B2Tau_CV_1_C2V_0_C3_1.root",
    "output_root/2022_VBFHHto2B2Tau_CV_1_C2V_1_C3_1.root",
    "output_root/2022_VBFHHto2B2Tau_CV_m1p21_C2V_1p94_C3_m0p94.root",
    "output_root/2022_VBFHHto2B2Tau_CV_m1p60_C2V_2p72_C3_m1p36.root"
]

# Colores para distinguir los histogramas
colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta]

# Nombre del árbol y de la rama
tree_name = "Event"
branch_name = "jet_0_pt"

# Crear canvas
c = ROOT.TCanvas("c", "Superpuestos", 800, 600)

# Lista para guardar los histogramas
histos = []

for i, file in enumerate(files):
    f = ROOT.TFile.Open(file)
    tree = f.Get(tree_name)
    
    hname = f"h{i}"
    h = ROOT.TH1F(hname, f"{branch_name} from {file}", 50, 0, 500)
    tree.Draw(f"{branch_name} >> {hname}", f"{branch_name} > 0", "goff")
    
    h.SetLineColor(colors[i])
    h.SetLineWidth(2)
    h.SetStats(0)
    h.Scale(1.0 / h.Integral())  # Normalización
    
    histos.append(h)

# Dibujar histogramas superpuestos
histos[0].Draw("HIST")
for h in histos[1:]:
    h.Draw("HIST SAME")

# Agregar leyenda
legend = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
for i, h in enumerate(histos):
    legend.AddEntry(h, files[i], "l")
legend.Draw()

# Guardar el canvas
c.SaveAs("jet_0_pt_superpuestos.png")
