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

void plotInputVariables_2024() {
    // Incluir pyInterface.h para funciones auxiliares
    gInterpreter->Declare("#include \"../include/pyInterface.h\"");
    
    std::vector<std::string> filenames = {
        "2024_GGHH_0.root", "2024_GGHH_6.root", "2024_GGHH_7.root",
        "2024_GGHH_13.root", "2024_GGHH_14.root", "2024_GGHH_15.root",
        "2024_GGHH_16.root", "2024_GGHH_17.root", "2024_GGHH_18.root"
    };

    std::string path = "/eos/user/e/emartinv/HHBtag_Training/training_skims_2024/";

    // Función para extraer el nodo del nombre del archivo
    auto getNodeFromFilename = [](const std::string& filename) -> std::string {
        // Extraer el número del final del nombre: 2024_GGHH_X.root -> X
        std::size_t start = filename.find_last_of("_") + 1;
        std::size_t end = filename.find(".root");
        if (start != std::string::npos && end != std::string::npos) {
            return filename.substr(start, end - start);
        }
        return "unknown";
    };

    // Mapear nodos específicos a colores específicos
    std::map<std::string, int> nodeToColor = {
        {"0", kRed},            // Rojo para nodo 0
        {"6", kGray+2},           // Azul para nodo 6
        {"7", kMagenta},        // Verde para nodo 7
        {"13", kBlue},       // Magenta para nodo 13
        {"14", kOrange},        // Naranja para nodo 14
        {"15", kCyan},          // Cian para nodo 15
        {"16", kViolet},        // Violeta para nodo 16
        {"17", kGreen+2},        // Gris para nodo 17
        {"18", kPink}          // Azul cielo para nodo 18
    };

    std::vector<int> colors = {kRed, kGray+2, kMagenta, kBlue, kOrange, kCyan, kViolet, kGreen+2, kPink};

    // Variables a plotear (definidas como en InputsProducer_2024.py)
    std::vector<std::string> variables_to_plot = {
        "jet_0_pt", "jet_1_pt", "jet_0_eta", "jet_1_eta", 
        "jet_0_E", "jet_1_E", "jet_0_M", "jet_1_M",
        "rel_jet_0_M_pt", "rel_jet_1_M_pt", "rel_jet_0_E_pt", "rel_jet_1_E_pt",
        "jet_0_deepFlavour", "jet_1_deepFlavour",
        "jet_0_ParticleNet", "jet_1_ParticleNet",
        "jet_0_UParT", "jet_1_UParT",
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
        {"jet_0_UParT", {50, 0.5}}, {"jet_1_UParT", {50, 0}},
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
        {"jet_0_UParT", 1}, {"jet_1_UParT", 1},
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
        {"jet_0_UParT", "Leading Jet UParT B-tag"}, {"jet_1_UParT", "Sub-leading Jet UParT B-tag"},
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
        {"jet_0_UParT", "UParT B-tag score"}, {"jet_1_UParT", "UParT B-tag score"},
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
    gSystem->mkdir("output_variables_24", kTRUE);

    // Loop sobre las variables definidas
    for (const auto& var : variables_to_plot) {
        std::string canvasName = var;
        std::string canvasTitle = varTitles[var];
        TCanvas* c = new TCanvas(canvasName.c_str(), canvasTitle.c_str(), 800, 600);
        TLegend* legend = new TLegend(0.55, 0.65, 0.88, 0.88);
        legend->SetTextSize(0.03);
        float maxY = 0, maxY_norm = 0;

        std::vector<TH1D*> histos, histos_norm;
        std::vector<TFile*> files;

        // Loop sobre archivos
        for (size_t i = 0; i < filenames.size(); ++i) {
            std::string file = filenames[i];
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
                                           .Define(Form("jet_%d_UParT", jet_idx), Form("jet_%d_valid ? RecoJet_btagUParTAK4B.at(%d) : 0.f", jet_idx, jet_idx))
                                           .Define(Form("jet_%d_htt_dphi", jet_idx), Form("jet_%d_valid ? ROOT::Math::VectorUtil::DeltaPhi(htt_p4, jet_%d_p4) : 0.f", jet_idx, jet_idx))
                                           .Define(Form("jet_%d_htt_deta", jet_idx), Form("jet_%d_valid ? (htt_p4.eta() - jet_%d_p4.eta()) : 0.f", jet_idx, jet_idx));
            }

            // Crear histogramas
            std::string hname = "h_" + var + "_" + std::to_string(i);
            std::string hname_norm = "h_" + var + "_norm_" + std::to_string(i);
            std::string htitle = varTitles[var] + ";" + xAxisTitles[var] + ";Events";
            std::string htitle_norm = varTitles[var] + " (norm);" + xAxisTitles[var] + ";Normalized Events";
            
            auto h = df_with_vars.Histo1D({hname.c_str(), htitle.c_str(), binning[var].first, static_cast<double>(binning[var].second), static_cast<double>(xmax[var])}, var);
            auto h_norm = df_with_vars.Histo1D({hname_norm.c_str(), htitle_norm.c_str(), binning[var].first, static_cast<double>(binning[var].second), static_cast<double>(xmax[var])}, var);
            
            // Obtener nodo del archivo y asignar color específico
            std::string node = getNodeFromFilename(file);
            int nodeColor = nodeToColor.count(node) ? nodeToColor[node] : colors[i % colors.size()];
            
            // Configurar histogramas
            h->SetLineColor(nodeColor);
            h->SetLineWidth(2);
            h->SetDirectory(0);
            
            h_norm->SetLineColor(nodeColor);
            h_norm->SetLineWidth(2);
            h_norm->SetDirectory(0);
            
            // Normalizar el histograma
            if (h_norm->Integral() > 0) {
                h_norm->Scale(1.0 / h_norm->Integral());
            }

            maxY = (maxY > h->GetMaximum()) ? maxY : h->GetMaximum();
            maxY_norm = (maxY_norm > h_norm->GetMaximum()) ? maxY_norm : h_norm->GetMaximum();
            
            // Clonar los histogramas para mantenerlos en memoria
            TH1D* h_clone = (TH1D*)h->Clone();
            TH1D* h_norm_clone = (TH1D*)h_norm->Clone();
            h_clone->SetDirectory(0);
            h_norm_clone->SetDirectory(0);
            
            // Quitar estadísticas
            h_clone->SetStats(0);
            h_norm_clone->SetStats(0);
            
            // Crear nombre para la leyenda: 2024_GGHH_X.root -> 24_GGF_X
            std::string legendName = file;
            if (legendName.find("2024_GGHH_") != std::string::npos) {
                legendName = legendName.replace(legendName.find("2024_GGHH_"), 10, "24_GGF_");
                if (legendName.find(".root") != std::string::npos) {
                    legendName = legendName.replace(legendName.find(".root"), 5, "");
                }
            }
            
            histos.push_back(h_clone);
            histos_norm.push_back(h_norm_clone);
            legend->AddEntry(h_clone, legendName.c_str(), "l");
            legend->SetMargin(0.2); // margen interno más pequeño
            legend->SetBorderSize(0);
            legend->SetFillStyle(0);
            legend->SetEntrySeparation(0.2); // reduce separación vertical
            legend->SetNColumns(1);          // opcional: varias columnas
            legend->SetTextSize(0.03);
            legend->SetFillColorAlpha(0, 0); // fondo transparente
            legend->SetBorderSize(0);
            legend->SetY1NDC(legend->GetY2NDC() - 0.04 * legend->GetNRows());
            files.push_back(f);
        }

        // Dibujar histogramas sin normalizar
        for (size_t i = 0; i < histos.size(); ++i) {
            if (histos[i] == nullptr) continue;
            histos[i]->SetMaximum(maxY * 1.2);
            histos[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
        }

        // Configurar escala logarítmica si es necesario
        if (shouldUseLogScale(var)) {
            c->SetLogy(1);
            // Ajustar el mínimo para variables b-tagging (evitar valores 0)
            for (size_t i = 0; i < histos.size(); ++i) {
                if (histos[i] == nullptr) continue;
                histos[i]->SetMinimum(0.1);
            }
        }

        legend->Draw();
        c->SaveAs(("output_variables_24/" + canvasName + ".png").c_str());
        c->SaveAs(("output_variables_24/" + canvasName + ".pdf").c_str());
        c->SaveAs(("output_variables_24/" + canvasName + ".root").c_str());

        // Crear canvas normalizado
        std::string canvasName_norm = canvasName + "_norm";
        TCanvas* c_norm = new TCanvas(canvasName_norm.c_str(), (canvasTitle + " (normalized)").c_str(), 800, 600);
        TLegend* legend_norm = new TLegend(0.55, 0.65, 0.88, 0.88);
        legend_norm->SetTextSize(0.03);

        // Dibujar histogramas normalizados
        for (size_t i = 0; i < histos_norm.size(); ++i) {
            if (histos_norm[i] == nullptr) continue;
            histos_norm[i]->SetMaximum(maxY_norm * 1.2);
            histos_norm[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
            
            // Crear nombre para la leyenda: 2024_GGHH_X.root -> 24_GGF_X
            std::string legendName = filenames[i];
            if (legendName.find("2024_GGHH_") != std::string::npos) {
                legendName = legendName.replace(legendName.find("2024_GGHH_"), 10, "24_GGF_");
                if (legendName.find(".root") != std::string::npos) {
                    legendName = legendName.replace(legendName.find(".root"), 5, "");
                }
            }
            
            legend_norm->AddEntry(histos_norm[i], legendName.c_str(), "l");
            legend_norm->SetMargin(0.2); // margen interno más pequeño
            legend_norm->SetBorderSize(0);
            legend_norm->SetFillStyle(0);
            legend_norm->SetEntrySeparation(0.2); // reduce separación vertical
            legend_norm->SetNColumns(1);          // opcional: varias columnas
            legend_norm->SetTextSize(0.03);
            legend_norm->SetFillColorAlpha(0, 0); // fondo transparente
            legend_norm->SetBorderSize(0);
            legend_norm->SetY1NDC(legend_norm->GetY2NDC() - 0.04 * legend_norm->GetNRows());
        }

        // Configurar escala logarítmica si es necesario para histogramas normalizados
        if (shouldUseLogScale(var)) {
            c_norm->SetLogy(1);
            // Ajustar el mínimo para variables b-tagging (evitar valores 0)
            for (size_t i = 0; i < histos_norm.size(); ++i) {
                if (histos_norm[i] == nullptr) continue;
                histos_norm[i]->SetMinimum(1e-5);  // Valor menor para histogramas normalizados
            }
        }

        legend_norm->Draw();
        c_norm->SaveAs(("output_variables_24/" + canvasName_norm + ".png").c_str());
        c_norm->SaveAs(("output_variables_24/" + canvasName_norm + ".pdf").c_str());
        c_norm->SaveAs(("output_variables_24/" + canvasName_norm + ".root").c_str());
    }
}   
