#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>

void plotInputVariables_2223() {
    // Incluir pyInterface.h para funciones auxiliares
    gInterpreter->Declare("#include \"../include/pyInterface.h\"");
    
    // Lista completa de archivos organizados por nodo
    std::vector<std::string> all_filenames = {
        // 2022
        "2022_GluGlutoHHto2B2Tau_6.root", "2022_GluGlutoHHto2B2Tau_7.root", 
        "2022_GluGlutoHHto2B2Tau_box.root", "2022_GluGlutoHHto2B2Tau_SM.root",
        "2022_VBFtoHH2B2Tau_1.root", "2022_VBFtoHH2B2Tau_2.root", "2022_VBFtoHH2B2Tau_3.root",
        "2022_VBFtoHH2B2Tau_4.root", "2022_VBFtoHH2B2Tau_5.root", "2022_VBFtoHH2B2Tau_6.root",
        "2022_VBFtoHH2B2Tau_7.root", "2022_VBFtoHH2B2Tau_8.root", "2022_VBFtoHH2B2Tau_9.root",
        "2022_VBFtoHH2B2Tau_SM.root",
        // 2022EE
        "2022EE_GluGlutoHHto2B2Tau_6.root", "2022EE_GluGlutoHHto2B2Tau_7.root",
        "2022EE_GluGlutoHHto2B2Tau_box.root", "2022EE_GluGlutoHHto2B2Tau_SM.root",
        "2022EE_VBFtoHH2B2Tau_1.root", "2022EE_VBFtoHH2B2Tau_2.root", "2022EE_VBFtoHH2B2Tau_3.root",
        "2022EE_VBFtoHH2B2Tau_4.root", "2022EE_VBFtoHH2B2Tau_5.root", "2022EE_VBFtoHH2B2Tau_6.root",
        "2022EE_VBFtoHH2B2Tau_7.root", "2022EE_VBFtoHH2B2Tau_8.root", "2022EE_VBFtoHH2B2Tau_9.root",
        "2022EE_VBFtoHH2B2Tau_SM.root",
        // 2023
        "2023_GluGlutoHHto2B2Tau_6.root", "2023_GluGlutoHHto2B2Tau_7.root",
        "2023_GluGlutoHHto2B2Tau_box.root", "2023_GluGlutoHHto2B2Tau_SM.root",
        "2023_VBFtoHH2B2Tau_1.root", "2023_VBFtoHH2B2Tau_2.root", "2023_VBFtoHH2B2Tau_3.root",
        "2023_VBFtoHH2B2Tau_4.root", "2023_VBFtoHH2B2Tau_5.root", "2023_VBFtoHH2B2Tau_6.root",
        "2023_VBFtoHH2B2Tau_7.root", "2023_VBFtoHH2B2Tau_8.root", "2023_VBFtoHH2B2Tau_9.root",
        "2023_VBFtoHH2B2Tau_SM.root",
        // 2023BPix
        "2023BPix_GluGlutoHHto2B2Tau_6.root", "2023BPix_GluGlutoHHto2B2Tau_7.root",
        "2023BPix_GluGlutoHHto2B2Tau_box.root", "2023BPix_GluGlutoHHto2B2Tau_SM.root",
        "2023BPix_VBFtoHH2B2Tau_1.root", "2023BPix_VBFtoHH2B2Tau_2.root", "2023BPix_VBFtoHH2B2Tau_3.root",
        "2023BPix_VBFtoHH2B2Tau_4.root", "2023BPix_VBFtoHH2B2Tau_5.root", "2023BPix_VBFtoHH2B2Tau_6.root",
        "2023BPix_VBFtoHH2B2Tau_7.root", "2023BPix_VBFtoHH2B2Tau_8.root", "2023BPix_VBFtoHH2B2Tau_9.root",
        "2023BPix_VBFtoHH2B2Tau_SM.root"
    };

    std::string path = "/eos/user/e/emartinv/HHBtag_Training/training_skims_Run3_v3.2/";

    // Función para extraer el nodo del nombre del archivo
    auto getNode = [](const std::string& filename) -> std::string {
        if (filename.find("_SM.root") != std::string::npos) return "SM";
        if (filename.find("_box.root") != std::string::npos) return "13";
        if (filename.find("_1.root") != std::string::npos) return "1";
        if (filename.find("_2.root") != std::string::npos) return "2";
        if (filename.find("_3.root") != std::string::npos) return "3";
        if (filename.find("_4.root") != std::string::npos) return "4";
        if (filename.find("_5.root") != std::string::npos) return "5";
        if (filename.find("_6.root") != std::string::npos) return "6";
        if (filename.find("_7.root") != std::string::npos) return "7";
        if (filename.find("_8.root") != std::string::npos) return "8";
        if (filename.find("_9.root") != std::string::npos) return "9";
        return "unknown";
    };

    // Función para determinar el tipo (GluGlu o VBF)
    auto getType = [](const std::string& filename) -> std::string {
        if (filename.find("GluGlutoHH") != std::string::npos) return "GGF";
        if (filename.find("VBFtoHH") != std::string::npos) return "VBF";
        return "unknown";
    };

    // Mapear nodos a números para la leyenda
    std::map<std::string, int> nodeToNumber = {
        {"SM", 0}, {"13", 13}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, 
        {"5", 5}, {"6", 6}, {"7", 7}, {"8", 8}, {"9", 9}
    };

    // Mapear nodos específicos a colores específicos
    std::map<std::string, int> nodeToColor = {
        {"SM", kRed},           // Rojo para SM (Standard Model)
        {"13", kBlue},         // Azul para box
        {"1", kGreen+2},        // Verde para nodo 1
        {"2", kAzure},        // Magenta para nodo 2
        {"3", kOrange},         // Naranja para nodo 3
        {"4", kCyan},           // Cian para nodo 4
        {"5", kViolet},         // Violeta para nodo 5
        {"6", kGray+2},         // Gris para nodo 6
        {"7", kMagenta},          // Azul cielo para nodo 7
        {"8", kPink},           // Rosa para nodo 8
        {"9", kSpring}          // Verde primavera para nodo 9
    };

    std::vector<int> colors = {kRed, kBlue, kGreen+2, kAzure, kOrange, kCyan, kViolet, kGray+2, kMagenta, kPink, kSpring, kTeal, kYellow+2, kOrange+2};

    // Variables a plotear (definidas como en InputsProducer_2024.py)
    std::vector<std::string> variables_to_plot = {
        "jet_0_pt", "jet_1_pt", "jet_0_eta", "jet_1_eta", 
        "jet_0_E", "jet_1_E", "jet_0_M", "jet_1_M",
        "rel_jet_0_M_pt", "rel_jet_1_M_pt", "rel_jet_0_E_pt", "rel_jet_1_E_pt",
        "jet_0_deepFlavour", "jet_1_deepFlavour",
        "jet_0_ParticleNet", "jet_1_ParticleNet",
        "jet_0_htt_dphi", "jet_1_htt_dphi", "jet_0_htt_deta", "jet_1_htt_deta",
        "htt_pt", "htt_eta", "htt_met_dphi", "rel_met_pt_htt_pt", "n_jets"
    };

    std::map<std::string, std::pair<int, int>> binning = {
        {"jet_0_pt", {50, 0}}, {"jet_1_pt", {50, 0}}, 
        {"jet_0_eta", {50, -3}}, {"jet_1_eta", {50, -3}}, 
        {"jet_0_E", {50, 0}}, {"jet_1_E", {50, 0}},
        {"jet_0_M", {50, 0}}, {"jet_1_M", {50, 0}},
        {"rel_jet_0_M_pt", {50, 0}}, {"rel_jet_1_M_pt", {50, 0}},
        {"rel_jet_0_E_pt", {50, 0}}, {"rel_jet_1_E_pt", {50, 0}},
        {"jet_0_deepFlavour", {50, 0.5}}, {"jet_1_deepFlavour", {50, 0}},
        {"jet_0_ParticleNet", {50, 0.5}}, {"jet_1_ParticleNet", {50, 0}},
        {"jet_0_htt_dphi", {50, -3.2}}, {"jet_1_htt_dphi", {50, -3.2}},
        {"jet_0_htt_deta", {50, -5}}, {"jet_1_htt_deta", {50, -5}},
        {"htt_pt", {50, 0}}, {"htt_eta", {50, -5}}, 
        {"htt_met_dphi", {50, -3.2}}, {"rel_met_pt_htt_pt", {50, 0}}, {"n_jets", {15, 0}}
    };

    std::map<std::string, float> xmax = {
        {"jet_0_pt", 500}, {"jet_1_pt", 500}, 
        {"jet_0_eta", 3}, {"jet_1_eta", 3}, 
        {"jet_0_E", 600}, {"jet_1_E", 600},
        {"jet_0_M", 100}, {"jet_1_M", 100},
        {"rel_jet_0_M_pt", 0.5}, {"rel_jet_1_M_pt", 0.5},
        {"rel_jet_0_E_pt", 5}, {"rel_jet_1_E_pt", 5},
        {"jet_0_deepFlavour", 1}, {"jet_1_deepFlavour", 1},
        {"jet_0_ParticleNet", 1}, {"jet_1_ParticleNet", 1},
        {"jet_0_htt_dphi", 3.2}, {"jet_1_htt_dphi", 3.2},
        {"jet_0_htt_deta", 5}, {"jet_1_htt_deta", 5},
        {"htt_pt", 500}, {"htt_eta", 5}, 
        {"htt_met_dphi", 3.2}, {"rel_met_pt_htt_pt", 3}, {"n_jets", 15}
    };

    // Títulos para variables y ejes
    std::map<std::string, std::string> varTitles = {
        {"jet_0_pt", "Leading Jet p_{T}"}, {"jet_1_pt", "Sub-leading Jet p_{T}"},
        {"jet_0_eta", "Leading Jet #eta"}, {"jet_1_eta", "Sub-leading Jet #eta"},
        {"rel_jet_0_M_pt", "Leading Jet M/p_{T}"}, {"rel_jet_1_M_pt", "Sub-leading Jet M/p_{T}"},
        {"rel_jet_0_E_pt", "Leading Jet E/p_{T}"}, {"rel_jet_1_E_pt", "Sub-leading Jet E/p_{T}"},
        {"jet_0_deepFlavour", "Leading Jet DeepFlav B-tag"}, {"jet_1_deepFlavour", "Sub-leading Jet DeepFlav B-tag"},
        {"jet_0_ParticleNet", "Leading Jet PNet B-tag"}, {"jet_1_ParticleNet", "Sub-leading Jet PNet B-tag"},
        {"jet_0_htt_dphi", "Leading Jet #Delta#phi(jet, H_{#tau#tau})"}, {"jet_1_htt_dphi", "Sub-leading Jet #Delta#phi(jet, H_{#tau#tau})"},
        {"jet_0_htt_deta", "Leading Jet #Delta#eta(jet, H_{#tau#tau})"}, {"jet_1_htt_deta", "Sub-leading Jet #Delta#eta(jet, H_{#tau#tau})"},
        {"htt_pt", "H_{#tau#tau} p_{T}"}, {"htt_eta", "H_{#tau#tau} #eta"}, 
        {"htt_met_dphi", "#Delta#phi(MET, H_{#tau#tau})"}, {"rel_met_pt_htt_pt", "MET/H_{#tau#tau} Scalar p_{T}"}, {"n_jets", "Number of Jets"}
    };

    std::map<std::string, std::string> xAxisTitles = {
        {"jet_0_pt", "p_{T} [GeV]"}, {"jet_1_pt", "p_{T} [GeV]"},
        {"jet_0_eta", "#eta"}, {"jet_1_eta", "#eta"},
        {"rel_jet_0_M_pt", "M/p_{T}"}, {"rel_jet_1_M_pt", "M/p_{T}"},
        {"rel_jet_0_E_pt", "E/p_{T}"}, {"rel_jet_1_E_pt", "E/p_{T}"},
        {"jet_0_deepFlavour", "DeepFlav B-tag score"}, {"jet_1_deepFlavour", "DeepFlav B-tag score"},
        {"jet_0_ParticleNet", "PNet B-tag score"}, {"jet_1_ParticleNet", "PNet B-tag score"},
        {"jet_0_htt_dphi", "#Delta#phi(jet, H_{#tau#tau})"}, {"jet_1_htt_dphi", "#Delta#phi(jet, H_{#tau#tau})"},
        {"jet_0_htt_deta", "#Delta#eta(jet, H_{#tau#tau})"}, {"jet_1_htt_deta", "#Delta#eta(jet, H_{#tau#tau})"},
        {"htt_pt", "p_{T} [GeV]"}, {"htt_eta", "#eta"}, 
        {"htt_met_dphi", "#Delta#phi(MET, H_{#tau#tau})"}, {"rel_met_pt_htt_pt", "MET / H_{#tau#tau} Scalar p_{T}"}, {"n_jets", "N_{jets}"}
    };

    // Función para determinar si una variable debe usar escala logarítmica
    auto shouldUseLogScale = [](const std::string& varName) {
        return (varName.find("deepFlavour") != std::string::npos ||
                varName.find("ParticleNet") != std::string::npos ||
                varName.find("UParT") != std::string::npos);
    };

    // Crear carpeta de salida
    gSystem->mkdir("output_variables_2223", kTRUE);

    // Función para crear leyenda con posicionamiento automático
    auto createSmartLegend = [](std::vector<TH1D*>& histos, TCanvas* canvas, const std::string& suffix = "") {
        if (histos.empty()) return (TLegend*)nullptr;
        
        canvas->cd();
        
        // Dibujar histogramas para calcular el área ocupada
        for (size_t i = 0; i < histos.size(); ++i) {
            if (histos[i] == nullptr) continue;
            histos[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
        }
        
        canvas->Update();
        
        // Posiciones posibles para la leyenda (x1, y1, x2, y2)
        std::vector<std::array<double, 4>> positions = {
            {0.65, 0.70, 0.88, 0.88}, // Esquina superior derecha (por defecto)
            {0.12, 0.70, 0.35, 0.88}, // Esquina superior izquierda
            {0.65, 0.12, 0.88, 0.30}, // Esquina inferior derecha
            {0.12, 0.12, 0.35, 0.30}, // Esquina inferior izquierda
            {0.45, 0.70, 0.68, 0.88}, // Centro superior
            {0.45, 0.12, 0.68, 0.30}  // Centro inferior
        };
        
        // Encontrar la mejor posición (la que tenga menos superposición)
        double minOverlap = 1e10;
        int bestPosition = 0;
        
        for (size_t pos = 0; pos < positions.size(); ++pos) {
            double overlap = 0;
            
            // Calcular superposición con histogramas
            for (auto* hist : histos) {
                if (!hist) continue;
                
                // Obtener coordenadas del pad
                double x1_pad = gPad->GetX1();
                double x2_pad = gPad->GetX2();
                double y1_pad = gPad->GetY1();
                double y2_pad = gPad->GetY2();
                
                // Convertir posición de la leyenda a coordenadas del histograma
                double leg_x1 = x1_pad + (x2_pad - x1_pad) * positions[pos][0];
                double leg_x2 = x1_pad + (x2_pad - x1_pad) * positions[pos][2];
                double leg_y1 = y1_pad + (y2_pad - y1_pad) * positions[pos][1];
                double leg_y2 = y1_pad + (y2_pad - y1_pad) * positions[pos][3];
                
                // Buscar bins en el área de la leyenda
                int bin1 = hist->FindBin(leg_x1);
                int bin2 = hist->FindBin(leg_x2);
                
                for (int bin = bin1; bin <= bin2; ++bin) {
                    double binContent = hist->GetBinContent(bin);
                    double binCenter = hist->GetBinCenter(bin);
                    double normalizedHeight = binContent / hist->GetMaximum();
                    
                    // Si el bin está dentro del área de la leyenda
                    if (binCenter >= leg_x1 && binCenter <= leg_x2) {
                        double relativeY = normalizedHeight * (y2_pad - y1_pad) + y1_pad;
                        if (relativeY >= leg_y1 && relativeY <= leg_y2) {
                            overlap += normalizedHeight;
                        }
                    }
                }
            }
            
            if (overlap < minOverlap) {
                minOverlap = overlap;
                bestPosition = pos;
            }
        }
        
        // Crear leyenda en la mejor posición
        TLegend* legend = new TLegend(positions[bestPosition][0], positions[bestPosition][1], 
                                     positions[bestPosition][2], positions[bestPosition][3]);
        legend->SetTextSize(0.03);
        legend->SetFillStyle(1001);
        legend->SetFillColor(kWhite);
        legend->SetBorderSize(1);
        
        return legend;
    };

    // Agrupar archivos por nodo y tipo
    std::map<std::string, std::vector<std::string>> groups; // key: "node_type", value: list of files
    for (const auto& file : all_filenames) {
        std::string node = getNode(file);
        std::string type = getType(file);
        std::string group_key = node + "_" + type;
        groups[group_key].push_back(file);
    }

    // Loop sobre las variables definidas
    for (const auto& var : variables_to_plot) {
        std::cout << "Processing variable: " << var << std::endl;
        
        // Crear canvas separados para GGF y VBF
        TCanvas* c_ggf = new TCanvas((var + "_GGF").c_str(), (varTitles[var] + " - GluGlu").c_str(), 800, 600);
        TCanvas* c_vbf = new TCanvas((var + "_VBF").c_str(), (varTitles[var] + " - VBF").c_str(), 800, 600);
        
        // Crear leyendas con posicionamiento automático (se posicionarán después)
        TLegend* legend_ggf = nullptr;
        TLegend* legend_vbf = nullptr;
        
        float maxY_ggf = 0, maxY_norm_ggf = 0;
        float maxY_vbf = 0, maxY_norm_vbf = 0;

        std::vector<TH1D*> histos_ggf, histos_norm_ggf;
        std::vector<TH1D*> histos_vbf, histos_norm_vbf;

        // Loop sobre grupos (node_type combinations)
        for (const auto& group : groups) {
            std::string group_key = group.first;
            auto files_in_group = group.second;
            
            std::string node = group_key.substr(0, group_key.find("_"));
            std::string type = group_key.substr(group_key.find("_") + 1);
            
            std::cout << "Processing group: " << group_key << " with " << files_in_group.size() << " files" << std::endl;
            
            // Crear histograma combinado para este grupo
            std::string hname = "h_" + var + "_" + group_key;
            std::string hname_norm = "h_" + var + "_norm_" + group_key;
            std::string htitle = varTitles[var] + ";" + xAxisTitles[var] + ";Events";
            std::string htitle_norm = varTitles[var] + " (norm);" + xAxisTitles[var] + ";Normalized Events";
            
            TH1D* h_combined = new TH1D(hname.c_str(), htitle.c_str(), binning[var].first, static_cast<double>(binning[var].second), static_cast<double>(xmax[var]));
            TH1D* h_norm_combined = new TH1D(hname_norm.c_str(), htitle_norm.c_str(), binning[var].first, static_cast<double>(binning[var].second), static_cast<double>(xmax[var]));
            
            h_combined->SetDirectory(0);
            h_norm_combined->SetDirectory(0);
            
            // Sumar histogramas de todos los archivos en este grupo
            for (const auto& file : files_in_group) {
                std::cout << "  Adding file: " << file << std::endl;
                TFile* f = TFile::Open((path + file).c_str());
                if (!f || f->IsZombie()) {
                    std::cout << "Error opening file: " << file << std::endl;
                    continue;
                }

                // Crear RDataFrame y definir variables como en InputsProducer_2024.py
                ROOT::RDataFrame df("Event", path + file);
                
                // Definir variables básicas
                auto df_with_vars = df.Define("n_jets", "RecoJet_pt.size()")
                                      .Define("htt_scalar_pt", "HttCandidate_leg0_pt + HttCandidate_leg1_pt")
                                      .Define("htt_p4", "getHTTp4(HttCandidate_leg0_pt, HttCandidate_leg0_eta, HttCandidate_leg0_phi, HttCandidate_leg0_mass, HttCandidate_leg1_pt, HttCandidate_leg1_eta, HttCandidate_leg1_phi, HttCandidate_leg1_mass)")
                                      .Define("htt_pt", "htt_p4.pt()")
                                      .Define("htt_eta", "htt_p4.eta()")
                                      .Define("rel_met_pt_htt_pt", "PuppiMET_pt / htt_scalar_pt")
                                      .Define("htt_met_dphi", "TVector2::Phi_mpi_pi(PuppiMET_phi - htt_p4.phi())");

                // Definir variables de jets como en InputsProducer_2024.py
                for (int jet_idx = 0; jet_idx < 2; ++jet_idx) {
                    df_with_vars = df_with_vars.Define(Form("jet_%d_valid", jet_idx), Form("static_cast<float>(%d) < RecoJet_pt.size()", jet_idx))
                                               .Define(Form("jet_%d_p4", jet_idx), Form("jet_%d_valid ? LorentzVectorM(RecoJet_pt.at(%d), RecoJet_eta.at(%d), RecoJet_phi.at(%d), RecoJet_mass.at(%d)) : LorentzVectorM()", jet_idx, jet_idx, jet_idx, jet_idx, jet_idx))
                                               .Define(Form("jet_%d_pt", jet_idx), Form("jet_%d_valid ? jet_%d_p4.pt() : 0.f", jet_idx, jet_idx))
                                               .Define(Form("jet_%d_eta", jet_idx), Form("jet_%d_valid ? jet_%d_p4.eta() : 0.f", jet_idx, jet_idx))
                                               .Define(Form("jet_%d_E", jet_idx), Form("jet_%d_valid ? jet_%d_p4.E() : 0.f", jet_idx, jet_idx))
                                               .Define(Form("jet_%d_M", jet_idx), Form("jet_%d_valid ? jet_%d_p4.M() : 0.f", jet_idx, jet_idx))
                                               .Define(Form("rel_jet_%d_M_pt", jet_idx), Form("jet_%d_valid ? jet_%d_p4.M() / jet_%d_p4.pt() : 0.f", jet_idx, jet_idx, jet_idx))
                                               .Define(Form("rel_jet_%d_E_pt", jet_idx), Form("jet_%d_valid ? jet_%d_p4.E() / jet_%d_p4.pt() : 0.f", jet_idx, jet_idx, jet_idx))
                                               .Define(Form("jet_%d_deepFlavour", jet_idx), Form("jet_%d_valid ? RecoJet_btagDeepFlavB.at(%d) : 0.f", jet_idx, jet_idx))
                                               .Define(Form("jet_%d_ParticleNet", jet_idx), Form("jet_%d_valid ? RecoJet_btagPNetB.at(%d) : 0.f", jet_idx, jet_idx))
                                               .Define(Form("jet_%d_htt_dphi", jet_idx), Form("jet_%d_valid ? ROOT::Math::VectorUtil::DeltaPhi(htt_p4, jet_%d_p4) : 0.f", jet_idx, jet_idx))
                                               .Define(Form("jet_%d_htt_deta", jet_idx), Form("jet_%d_valid ? (htt_p4.eta() - jet_%d_p4.eta()) : 0.f", jet_idx, jet_idx));
                }

                // Crear histograma temporal para este archivo
                std::string temp_hname = "temp_h_" + var + "_" + file;
                auto h_temp = df_with_vars.Histo1D({temp_hname.c_str(), htitle.c_str(), binning[var].first, static_cast<double>(binning[var].second), static_cast<double>(xmax[var])}, var);
                
                // Añadir al histograma combinado
                h_combined->Add(h_temp.GetPtr());
                h_norm_combined->Add(h_temp.GetPtr());
                
                f->Close();
                delete f;
            }
            
            // Configurar histogramas del grupo
            int color = nodeToColor[node];  // Usar color específico del nodo
            h_combined->SetLineColor(color);
            h_combined->SetLineWidth(2);
            h_combined->SetStats(0);
            
            h_norm_combined->SetLineColor(color);
            h_norm_combined->SetLineWidth(2);
            h_norm_combined->SetStats(0);
            
            // Normalizar
            if (h_norm_combined->Integral() > 0) {
                h_norm_combined->Scale(1.0 / h_norm_combined->Integral());
            }
            
            // Crear nombre para la leyenda
            std::string legendName = "2223_" + type + "_" + std::to_string(nodeToNumber[node]);
            
            // Agregar a la colección apropiada (sin añadir a leyenda aún)
            if (type == "GGF") {
                histos_ggf.push_back(h_combined);
                histos_norm_ggf.push_back(h_norm_combined);
                
                maxY_ggf = (maxY_ggf > h_combined->GetMaximum()) ? maxY_ggf : h_combined->GetMaximum();
                maxY_norm_ggf = (maxY_norm_ggf > h_norm_combined->GetMaximum()) ? maxY_norm_ggf : h_norm_combined->GetMaximum();
            } else {
                histos_vbf.push_back(h_combined);
                histos_norm_vbf.push_back(h_norm_combined);
                
                maxY_vbf = (maxY_vbf > h_combined->GetMaximum()) ? maxY_vbf : h_combined->GetMaximum();
                maxY_norm_vbf = (maxY_norm_vbf > h_norm_combined->GetMaximum()) ? maxY_norm_vbf : h_norm_combined->GetMaximum();
            }
        }
        
        // Crear y configurar leyendas con posicionamiento automático
        legend_ggf = createSmartLegend(histos_ggf, c_ggf);
        legend_vbf = createSmartLegend(histos_vbf, c_vbf);
        
        // Añadir entradas a las leyendas
        for (const auto& group : groups) {
            std::string group_key = group.first;
            std::string node = group_key.substr(0, group_key.find("_"));
            std::string type = group_key.substr(group_key.find("_") + 1);
            std::string legendName = "2223_" + type + "_" + std::to_string(nodeToNumber[node]);
            int node_color = nodeToColor[node];  // Obtener color específico del nodo
            
            if (type == "GGF" && legend_ggf) {
                // Encontrar el histograma correspondiente
                for (auto* hist : histos_ggf) {
                    if (hist && hist->GetLineColor() == node_color) {
                        legend_ggf->AddEntry(hist, legendName.c_str(), "l");
                        legend_ggf->SetMargin(0.2); // margen interno más pequeño
                        legend_ggf->SetBorderSize(0);
                        legend_ggf->SetFillStyle(0);
                        legend_ggf->SetEntrySeparation(0.2); // reduce separación vertical
                        legend_ggf->SetNColumns(1);          // opcional: varias columnas
                        legend_ggf->SetTextSize(0.03);
                        legend_ggf->SetFillColorAlpha(0, 0); // fondo transparente
                        legend_ggf->SetBorderSize(0);
                        legend_ggf->SetY1NDC(legend_ggf->GetY2NDC() - 0.04 * legend_ggf->GetNRows());
                        break;
                    }
                }
            } else if (type == "VBF" && legend_vbf) {
                // Encontrar el histograma correspondiente
                for (auto* hist : histos_vbf) {
                    if (hist && hist->GetLineColor() == node_color) {
                        legend_vbf->AddEntry(hist, legendName.c_str(), "l");
                        legend_vbf->SetMargin(0.2); // margen interno más pequeño
                        legend_vbf->SetBorderSize(0);
                        legend_vbf->SetFillStyle(0);
                        legend_vbf->SetEntrySeparation(0.2); // reduce separación vertical
                        legend_vbf->SetNColumns(1);          // opcional: varias columnas
                        legend_vbf->SetTextSize(0.03);
                        legend_vbf->SetFillColorAlpha(0, 0); // fondo transparente
                        legend_vbf->SetBorderSize(0);
                        legend_vbf->SetY1NDC(legend_vbf->GetY2NDC() - 0.04 * legend_vbf->GetNRows());
                        break;
                    }
                }
            }
        }

        // Dibujar histogramas GGF sin normalizar
        c_ggf->cd();
        for (size_t i = 0; i < histos_ggf.size(); ++i) {
            if (histos_ggf[i] == nullptr) continue;
            histos_ggf[i]->SetMaximum(maxY_ggf * 1.2);
            histos_ggf[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
        }
        
        // Configurar escala logarítmica si es necesario
        if (shouldUseLogScale(var)) {
            c_ggf->SetLogy(1);
            // Ajustar el mínimo para variables b-tagging (evitar valores 0)
            for (size_t i = 0; i < histos_ggf.size(); ++i) {
                if (histos_ggf[i] == nullptr) continue;
                histos_ggf[i]->SetMinimum(0.1);
            }
        }
        
        if (legend_ggf) legend_ggf->Draw();
        c_ggf->SaveAs(("output_variables_2223/" + var + "_GGF.png").c_str());
        c_ggf->SaveAs(("output_variables_2223/" + var + "_GGF.pdf").c_str());
        c_ggf->SaveAs(("output_variables_2223/" + var + "_GGF.root").c_str());

        // Dibujar histogramas VBF sin normalizar
        c_vbf->cd();
        for (size_t i = 0; i < histos_vbf.size(); ++i) {
            if (histos_vbf[i] == nullptr) continue;
            histos_vbf[i]->SetMaximum(maxY_vbf * 1.2);
            histos_vbf[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
        }
        
        // Configurar escala logarítmica si es necesario
        if (shouldUseLogScale(var)) {
            c_vbf->SetLogy(1);
            // Ajustar el mínimo para variables b-tagging (evitar valores 0)
            for (size_t i = 0; i < histos_vbf.size(); ++i) {
                if (histos_vbf[i] == nullptr) continue;
                histos_vbf[i]->SetMinimum(0.1);
            }
        }
        
        if (legend_vbf) legend_vbf->Draw();
        c_vbf->SaveAs(("output_variables_2223/" + var + "_VBF.png").c_str());
        c_vbf->SaveAs(("output_variables_2223/" + var + "_VBF.pdf").c_str());
        c_vbf->SaveAs(("output_variables_2223/" + var + "_VBF.root").c_str());

        // Crear canvas normalizados
        TCanvas* c_ggf_norm = new TCanvas((var + "_GGF_norm").c_str(), (varTitles[var] + " - GluGlu (normalized)").c_str(), 800, 600);
        TCanvas* c_vbf_norm = new TCanvas((var + "_VBF_norm").c_str(), (varTitles[var] + " - VBF (normalized)").c_str(), 800, 600);
        
        // Crear leyendas con posicionamiento automático para histogramas normalizados
        TLegend* legend_ggf_norm = createSmartLegend(histos_norm_ggf, c_ggf_norm, "_norm");
        TLegend* legend_vbf_norm = createSmartLegend(histos_norm_vbf, c_vbf_norm, "_norm");
        
        // Añadir entradas a las leyendas normalizadas
        for (const auto& group : groups) {
            std::string group_key = group.first;
            std::string node = group_key.substr(0, group_key.find("_"));
            std::string type = group_key.substr(group_key.find("_") + 1);
            std::string legendName = "2223_" + type + "_" + std::to_string(nodeToNumber[node]);
            int node_color = nodeToColor[node];  // Obtener color específico del nodo
            
            if (type == "GGF" && legend_ggf_norm) {
                // Encontrar el histograma correspondiente
                for (auto* hist : histos_norm_ggf) {
                    if (hist && hist->GetLineColor() == node_color) {
                        legend_ggf_norm->AddEntry(hist, legendName.c_str(), "l");
                        legend_ggf_norm->SetMargin(0.2); // margen interno más pequeño
                        legend_ggf_norm->SetBorderSize(0);
                        legend_ggf_norm->SetFillStyle(0);
                        legend_ggf_norm->SetEntrySeparation(0.2); // reduce separación vertical
                        legend_ggf_norm->SetNColumns(1);          // opcional: varias columnas
                        legend_ggf_norm->SetTextSize(0.03);
                        legend_ggf_norm->SetFillColorAlpha(0, 0); // fondo transparente
                        legend_ggf_norm->SetBorderSize(0);
                        legend_ggf_norm->SetY1NDC(legend_ggf_norm->GetY2NDC() - 0.04 * legend_ggf_norm->GetNRows());
                        break;
                    }
                }
            } else if (type == "VBF" && legend_vbf_norm) {
                // Encontrar el histograma correspondiente
                for (auto* hist : histos_norm_vbf) {
                    if (hist && hist->GetLineColor() == node_color) {
                        legend_vbf_norm->AddEntry(hist, legendName.c_str(), "l");
                        legend_vbf_norm->SetMargin(0.2); // margen interno más pequeño
                        legend_vbf_norm->SetBorderSize(0);
                        legend_vbf_norm->SetFillStyle(0);
                        legend_vbf_norm->SetEntrySeparation(0.2); // reduce separación vertical
                        legend_vbf_norm->SetNColumns(1);          // opcional: varias columnas
                        legend_vbf_norm->SetTextSize(0.03);
                        legend_vbf_norm->SetFillColorAlpha(0, 0); // fondo transparente
                        legend_vbf_norm->SetBorderSize(0);
                        legend_vbf_norm->SetY1NDC(legend_vbf_norm->GetY2NDC() - 0.04 * legend_vbf_norm->GetNRows());
                        break;
                    }
                }
            }
        }

        // Dibujar histogramas GGF normalizados
        c_ggf_norm->cd();
        for (size_t i = 0; i < histos_norm_ggf.size(); ++i) {
            if (histos_norm_ggf[i] == nullptr) continue;
            histos_norm_ggf[i]->SetMaximum(maxY_norm_ggf * 1.2);
            histos_norm_ggf[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
        }
        
        // Configurar escala logarítmica si es necesario para histogramas normalizados
        if (shouldUseLogScale(var)) {
            c_ggf_norm->SetLogy(1);
            // Ajustar el mínimo para variables b-tagging (evitar valores 0)
            for (size_t i = 0; i < histos_norm_ggf.size(); ++i) {
                if (histos_norm_ggf[i] == nullptr) continue;
                histos_norm_ggf[i]->SetMinimum(1e-5);  // Valor menor para histogramas normalizados
            }
        }
        
        if (legend_ggf_norm) legend_ggf_norm->Draw();
        c_ggf_norm->SaveAs(("output_variables_2223/" + var + "_GGF_norm.png").c_str());
        c_ggf_norm->SaveAs(("output_variables_2223/" + var + "_GGF_norm.pdf").c_str());
        c_ggf_norm->SaveAs(("output_variables_2223/" + var + "_GGF_norm.root").c_str());

        // Dibujar histogramas VBF normalizados
        c_vbf_norm->cd();
        for (size_t i = 0; i < histos_norm_vbf.size(); ++i) {
            if (histos_norm_vbf[i] == nullptr) continue;
            histos_norm_vbf[i]->SetMaximum(maxY_norm_vbf * 1.2);
            histos_norm_vbf[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
        }
        
        // Configurar escala logarítmica si es necesario para histogramas normalizados
        if (shouldUseLogScale(var)) {
            c_vbf_norm->SetLogy(1);
            // Ajustar el mínimo para variables b-tagging (evitar valores 0)
            for (size_t i = 0; i < histos_norm_vbf.size(); ++i) {
                if (histos_norm_vbf[i] == nullptr) continue;
                histos_norm_vbf[i]->SetMinimum(1e-5);  // Valor menor para histogramas normalizados
            }
        }
        
        if (legend_vbf_norm) legend_vbf_norm->Draw();
        c_vbf_norm->SaveAs(("output_variables_2223/" + var + "_VBF_norm.png").c_str());
        c_vbf_norm->SaveAs(("output_variables_2223/" + var + "_VBF_norm.pdf").c_str());
        c_vbf_norm->SaveAs(("output_variables_2223/" + var + "_VBF_norm.root").c_str());
    }
}   
