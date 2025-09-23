#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPad.h>
#include <TLine.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>

void compare222324inputVariables() {
    // Incluir pyInterface.h para funciones auxiliares
    gInterpreter->Declare("#include \"../include/pyInterface.h\"");
    
    // Archivos de 2022-2023 (solo GGF)
    std::vector<std::string> files_2223 = {
        // 2022
        "2022_GluGlutoHHto2B2Tau_6.root", "2022_GluGlutoHHto2B2Tau_7.root", 
        "2022_GluGlutoHHto2B2Tau_box.root", "2022_GluGlutoHHto2B2Tau_SM.root",
        // 2022EE
        "2022EE_GluGlutoHHto2B2Tau_6.root", "2022EE_GluGlutoHHto2B2Tau_7.root",
        "2022EE_GluGlutoHHto2B2Tau_box.root", "2022EE_GluGlutoHHto2B2Tau_SM.root",
        // 2023
        "2023_GluGlutoHHto2B2Tau_6.root", "2023_GluGlutoHHto2B2Tau_7.root",
        "2023_GluGlutoHHto2B2Tau_box.root", "2023_GluGlutoHHto2B2Tau_SM.root",
        // 2023BPix
        "2023BPix_GluGlutoHHto2B2Tau_6.root", "2023BPix_GluGlutoHHto2B2Tau_7.root",
        "2023BPix_GluGlutoHHto2B2Tau_box.root", "2023BPix_GluGlutoHHto2B2Tau_SM.root"
    };

    // Archivos de 2024 (solo nodos específicos: 0=SM, 6, 7, 13=box)
    std::vector<std::string> files_2024 = {
        "2024_GGHH_0.root",   // SM
        "2024_GGHH_6.root",   // nodo 6
        "2024_GGHH_7.root",   // nodo 7
        "2024_GGHH_13.root"   // box
    };

    std::string path_2223 = "/eos/user/e/emartinv/HHBtag_Training/training_skims_Run3_v3.2/";
    std::string path_2024 = "/eos/user/e/emartinv/HHBtag_Training/training_skims_2024/";

    // Función para extraer el nodo del nombre del archivo
    auto getNode = [](const std::string& filename) -> std::string {
        if (filename.find("_SM.root") != std::string::npos) return "SM";
        if (filename.find("_box.root") != std::string::npos) return "13";
        if (filename.find("_0.root") != std::string::npos) return "SM";
        if (filename.find("_6.root") != std::string::npos) return "6";
        if (filename.find("_7.root") != std::string::npos) return "7";
        if (filename.find("_13.root") != std::string::npos) return "13";
        return "unknown";
    };

    // Mapear nodos específicos a colores específicos (consistente con otros scripts)
    std::map<std::string, int> nodeToColor = {
        {"SM", kRed},           // Rojo para SM (Standard Model)
        {"13", kBlue},         // Azul para box (nodo 13)
        {"6", kGray+2},        // Gris para nodo 6
        {"7", kMagenta}         // Magenta para nodo 7
    };

    // Variables a plotear (las mismas que en plotInputVariables_2223.C)
    std::vector<std::string> variables_to_plot = {
        "jet_0_pt", "jet_1_pt", "jet_0_eta", "jet_1_eta", 
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
    gSystem->mkdir("comparison_output_variables_222324", kTRUE);

    // Nodos que vamos a comparar
    std::vector<std::string> nodes_to_compare = {"SM", "6", "7", "13"};

    // Loop sobre variables
    for (const auto& var : variables_to_plot) {
        std::cout << "Processing variable: " << var << std::endl;
        
        // Loop sobre nodos
        for (const auto& node : nodes_to_compare) {
            std::cout << "  Processing node: " << node << std::endl;
            
            
            std::string canvasName = var + "_node_" + node + "_comparison";
            std::string canvasTitle = varTitles[var] + " - Node " + node;
            TCanvas* c = new TCanvas(canvasName.c_str(), canvasTitle.c_str(), 900, 800);
            
            // Dividir el canvas en dos pads: distribución arriba y ratio abajo
            TPad* pad1 = new TPad("pad1", "Distribution pad", 0, 0.3, 1, 1.0);
            TPad* pad2 = new TPad("pad2", "Ratio pad", 0, 0.0, 1, 0.3);
            
            pad1->SetBottomMargin(0.02);  // Sin espacio entre pads
            pad2->SetTopMargin(0.02);
            pad2->SetBottomMargin(0.3);   // Espacio para labels del eje x
            
            pad1->Draw();
            pad2->Draw();
            
            // Crear histogramas combinados
            std::string h2223_name = "h_2223_" + var + "_node_" + node;
            std::string h2024_name = "h_2024_" + var + "_node_" + node;
            std::string h2223_norm_name = "h_2223_norm_" + var + "_node_" + node;
            std::string h2024_norm_name = "h_2024_norm_" + var + "_node_" + node;
            
            std::string htitle = varTitles[var] + " - Node " + node + ";;" + "Normalized Events";  // Sin título en x para pad superior
            std::string htitle_ratio = ";" + xAxisTitles[var] + ";Ratio 2024/22&23";  // Sin título principal para ratio
            
            TH1D* h_2223_norm = new TH1D(h2223_norm_name.c_str(), htitle.c_str(), binning[var].first, static_cast<double>(binning[var].second), static_cast<double>(xmax[var]));
            TH1D* h_2024_norm = new TH1D(h2024_norm_name.c_str(), htitle.c_str(), binning[var].first, static_cast<double>(binning[var].second), static_cast<double>(xmax[var]));
            
            h_2223_norm->SetDirectory(0);
            h_2024_norm->SetDirectory(0);
            
            // Procesar archivos 2022-2023
            for (const auto& file : files_2223) {
                std::string file_node = getNode(file);
                if (file_node != node) continue;
                
                std::cout << "    Adding 2223 file: " << file << std::endl;
                TFile* f = TFile::Open((path_2223 + file).c_str());
                if (!f || f->IsZombie()) {
                    std::cout << "Error opening file: " << file << std::endl;
                    continue;
                }

                ROOT::RDataFrame df("Event", path_2223 + file);
                
                // Definir variables básicas
                auto df_with_vars = df.Define("n_jets", "RecoJet_pt.size()")
                                      .Define("htt_scalar_pt", "HttCandidate_leg0_pt + HttCandidate_leg1_pt")
                                      .Define("htt_p4", "getHTTp4(HttCandidate_leg0_pt, HttCandidate_leg0_eta, HttCandidate_leg0_phi, HttCandidate_leg0_mass, HttCandidate_leg1_pt, HttCandidate_leg1_eta, HttCandidate_leg1_phi, HttCandidate_leg1_mass)")
                                      .Define("htt_pt", "htt_p4.pt()")
                                      .Define("htt_eta", "htt_p4.eta()")
                                      .Define("rel_met_pt_htt_pt", "PuppiMET_pt / htt_scalar_pt")
                                      .Define("htt_met_dphi", "TVector2::Phi_mpi_pi(PuppiMET_phi - htt_p4.phi())");

                // Definir variables de jets
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

                // Crear histograma temporal
                std::string temp_hname = "temp_h_2223_" + var + "_" + file;
                auto h_temp = df_with_vars.Histo1D({temp_hname.c_str(), htitle.c_str(), binning[var].first, static_cast<double>(binning[var].second), static_cast<double>(xmax[var])}, var);
                
                // Añadir al histograma combinado normalizado
                h_2223_norm->Add(h_temp.GetPtr());
                
                f->Close();
                delete f;
            }
            
            // Procesar archivos 2024
            for (const auto& file : files_2024) {
                std::string file_node = getNode(file);
                if (file_node != node) continue;
                
                std::cout << "    Adding 2024 file: " << file << std::endl;
                TFile* f = TFile::Open((path_2024 + file).c_str());
                if (!f || f->IsZombie()) {
                    std::cout << "Error opening file: " << file << std::endl;
                    continue;
                }

                ROOT::RDataFrame df("Event", path_2024 + file);
                
                // Definir variables básicas (igual que arriba)
                auto df_with_vars = df.Define("n_jets", "RecoJet_pt.size()")
                                      .Define("htt_scalar_pt", "HttCandidate_leg0_pt + HttCandidate_leg1_pt")
                                      .Define("htt_p4", "getHTTp4(HttCandidate_leg0_pt, HttCandidate_leg0_eta, HttCandidate_leg0_phi, HttCandidate_leg0_mass, HttCandidate_leg1_pt, HttCandidate_leg1_eta, HttCandidate_leg1_phi, HttCandidate_leg1_mass)")
                                      .Define("htt_pt", "htt_p4.pt()")
                                      .Define("htt_eta", "htt_p4.eta()")
                                      .Define("rel_met_pt_htt_pt", "PuppiMET_pt / htt_scalar_pt")
                                      .Define("htt_met_dphi", "TVector2::Phi_mpi_pi(PuppiMET_phi - htt_p4.phi())");

                // Definir variables de jets
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
                                               .Define(Form("jet_%d_UParT", jet_idx), Form("jet_%d_valid ? RecoJet_btagUParTAK4B.at(%d) : 0.f", jet_idx, jet_idx))
                                               .Define(Form("jet_%d_htt_dphi", jet_idx), Form("jet_%d_valid ? ROOT::Math::VectorUtil::DeltaPhi(htt_p4, jet_%d_p4) : 0.f", jet_idx, jet_idx))
                                               .Define(Form("jet_%d_htt_deta", jet_idx), Form("jet_%d_valid ? (htt_p4.eta() - jet_%d_p4.eta()) : 0.f", jet_idx, jet_idx));
                }

                // Crear histograma temporal
                std::string temp_hname = "temp_h_2024_" + var + "_" + file;
                auto h_temp = df_with_vars.Histo1D({temp_hname.c_str(), htitle.c_str(), binning[var].first, static_cast<double>(binning[var].second), static_cast<double>(xmax[var])}, var);
                
                // Añadir al histograma combinado normalizado
                h_2024_norm->Add(h_temp.GetPtr());
                
                f->Close();
                delete f;
            }
            
            // Configurar colores y estilos
            int node_color = nodeToColor[node];
            h_2223_norm->SetLineColor(node_color);
            h_2223_norm->SetLineWidth(3);
            h_2223_norm->SetLineStyle(1);  // Línea sólida para 2022-23
            h_2223_norm->SetStats(0);
            
            h_2024_norm->SetLineColor(node_color);
            h_2024_norm->SetLineWidth(3);
            h_2024_norm->SetLineStyle(2);  // Línea discontinua para 2024
            h_2024_norm->SetStats(0);
            
            // Normalizar histogramas
            if (h_2223_norm->Integral() > 0) h_2223_norm->Scale(1.0 / h_2223_norm->Integral());
            if (h_2024_norm->Integral() > 0) h_2024_norm->Scale(1.0 / h_2024_norm->Integral());
            
            // Encontrar máximo para escalar
            float maxY_norm = std::max(h_2223_norm->GetMaximum(), h_2024_norm->GetMaximum());
            
            // Crear histograma de ratio
            TH1D* h_ratio = (TH1D*)h_2024_norm->Clone("h_ratio");
            h_ratio->SetTitle(htitle_ratio.c_str());
            h_ratio->Divide(h_2223_norm);
            h_ratio->SetLineColor(node_color);
            h_ratio->SetLineWidth(3);
            h_ratio->SetStats(0);
            h_ratio->SetMinimum(0.5);
            h_ratio->SetMaximum(1.5);
            
            // Dibujar distribuciones normalizadas en pad superior
            pad1->cd();
            h_2223_norm->SetMaximum(maxY_norm * 1.2);
            h_2223_norm->GetXaxis()->SetLabelSize(0);  // No mostrar labels en x
            h_2223_norm->GetXaxis()->SetTitleSize(0);  // No mostrar título en x
            h_2223_norm->GetYaxis()->SetTitle("Normalized Events");
            h_2223_norm->Draw("HIST");
            h_2024_norm->Draw("HIST SAME");
            
            // Configurar escala logarítmica si es necesario
            if (shouldUseLogScale(var)) {
                pad1->SetLogy(1);
                h_2223_norm->SetMinimum(1e-5);
                h_2024_norm->SetMinimum(1e-5);
            }
            
            // Crear leyenda
            TLegend* legend = new TLegend(0.65, 0.75, 0.88, 0.88);
            legend->AddEntry(h_2223_norm, ("22&23 Node " + node).c_str(), "l");
            legend->AddEntry(h_2024_norm, ("2024 Node " + node).c_str(), "l");
            legend->SetTextSize(0.04);
            legend->SetFillStyle(0);
            legend->SetBorderSize(0);
            legend->Draw();
            
            // Dibujar ratio en pad inferior
            pad2->cd();
            h_ratio->GetYaxis()->SetTitle("Ratio 2024/22&23");
            h_ratio->GetYaxis()->SetTitleSize(0.08);
            h_ratio->GetYaxis()->SetTitleOffset(0.5);
            h_ratio->GetYaxis()->SetLabelSize(0.06);
            h_ratio->GetXaxis()->SetTitle(xAxisTitles[var].c_str());
            h_ratio->GetXaxis()->SetTitleSize(0.08);
            h_ratio->GetXaxis()->SetLabelSize(0.06);
            h_ratio->Draw("HIST");
            
            // Línea horizontal en 1
            TLine* line = new TLine(binning[var].second, 1.0, xmax[var], 1.0);
            line->SetLineColor(kBlack);
            line->SetLineStyle(2);
            line->Draw();
            
            // Guardar plots
            c->SaveAs(("comparison_output_variables_222324/" + canvasName + ".png").c_str());
            c->SaveAs(("comparison_output_variables_222324/" + canvasName + ".pdf").c_str());
            c->SaveAs(("comparison_output_variables_222324/" + canvasName + ".root").c_str());
        }
        
    }
    
    std::cout << "Comparison complete! Plots saved in comparison_output_variables_222324/" << std::endl;
}