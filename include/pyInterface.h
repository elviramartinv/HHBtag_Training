/*! Definition of functions for calulating most used quantities
*/
#pragma once


using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
using LorentzVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
using LorentzVectorE = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double>>;
using RVecI = ROOT::VecOps::RVec<int>;
using RVecS = ROOT::VecOps::RVec<size_t>;
using RVecUC = ROOT::VecOps::RVec<UChar_t>;
using RVecF = ROOT::VecOps::RVec<float>;
using RVecB = ROOT::VecOps::RVec<bool>;
using RVecVecI = ROOT::VecOps::RVec<RVecI>;
using RVecLV = ROOT::VecOps::RVec<LorentzVectorM>;
using RVecSetInt = ROOT::VecOps::RVec<std::set<int>>;

LorentzVectorM getHTTp4(float pt0, float eta0, float phi0, float mass0,
                            float pt1, float eta1, float phi1, float mass1)
{
    LorentzVectorM lep_p40 = LorentzVectorM(pt0, eta0, phi0, mass0);
    LorentzVectorM lep_p41 = LorentzVectorM(pt1, eta1, phi1, mass1);
    return lep_p40 + lep_p41;
}



// LorentzVectorM getHTTp4 (const ROOT::VecOps::RVec<LorentzVectorM>& lep_p4, const ROOT::VecOps::RVec<int>& signal_tau_idx)
// {
    // return lep_p4.at(signal_tau_idx.at(0)) + lep_p4.at(signal_tau_idx.at(1));
// }

float getHTTScalarPt (const ROOT::VecOps::RVec<LorentzVectorM>& lep_p4, const ROOT::VecOps::RVec<int>& signal_tau_idx)
{
    return lep_p4.at(signal_tau_idx.at(0)).Pt() + lep_p4.at(signal_tau_idx.at(1)).Pt();
}

ROOT::VecOps::RVec<int> getSignalTauIndices_Gen(const ROOT::VecOps::RVec<LorentzVectorM>& lep_p4,
                                                const ROOT::VecOps::RVec<int>& lep_genTauIndex)
{
    ROOT::VecOps::RVec<int> indexes;
    int n_tau = 0;
    for(size_t n = 0; n < lep_p4.size(); ++n) {
        if(lep_genTauIndex.at(n) >= 0) {
            indexes.push_back(n);
            n_tau++;
        }
    }
    if(n_tau != 2)
        throw std::runtime_error("too few taus");
    return indexes;
}

ROOT::VecOps::RVec<int> getSignalTauIndicesDeep_Tau(const ROOT::VecOps::RVec<LorentzVectorM>& lep_p4,
                                                    const ROOT::VecOps::RVec<float>& byDeepTau2017v2p1VSjet,
                                                    const ROOT::VecOps::RVec<float>& byDeepTau2017v2p1VSeraw,
                                                    const ROOT::VecOps::RVec<float>& byDeepTau2017v2p1VSmuraw,
                                                    const ROOT::VecOps::RVec<int>& lep_type,
                                                    int channelId)
{
    ROOT::VecOps::RVec<size_t> ordered_index;
    ROOT::VecOps::RVec<size_t> selected_taus_indices;
    // values taken from: https://github.com/cms-sw/cmssw/blob/master/RecoTauTag/RecoTau/python/tools/runTauIdMVA.py#L658-L685
    double  losser_wp_vs_e = 0.0630386;
    double  losser_wp_vs_mu = 0.1058354;
    static constexpr size_t light_lepton_idx = 0;
    for(size_t lep_index = 0; lep_index < lep_p4.size(); ++lep_index){
        if(lep_type.at(lep_index) == 2 && (byDeepTau2017v2p1VSeraw.at(lep_index) < losser_wp_vs_e ||
           byDeepTau2017v2p1VSmuraw.at(lep_index) < losser_wp_vs_mu )) continue;
        if(lep_type.at(lep_index) == 2 && channelId != 2 && ROOT::Math::VectorUtil::DeltaR(lep_p4.at(lep_index),
                                                                                            lep_p4.at(light_lepton_idx)) < 0.1) continue;
        ordered_index.push_back(lep_index);
    }

    std::sort(ordered_index.begin(), ordered_index.end(), [&](size_t a, size_t b){
        if(lep_type.at(a) ==  2 && lep_type.at(b) ==  2)
            return byDeepTau2017v2p1VSjet.at(a) > byDeepTau2017v2p1VSjet.at(b);
        return (lep_type.at(a) < lep_type.at(b));
    });

    for(size_t n = 0; n < std::min<size_t>(ordered_index.size(), 2); ++n)
        selected_taus_indices.push_back(ordered_index.at(n));
    return selected_taus_indices;

}

float HTTScalarPt (const ROOT::VecOps::RVec<LorentzVectorM>& lep_p4, const ROOT::VecOps::RVec<int>& lep_genTauIndex)
{
    size_t n_tau = 0;
    float h_tautau_pt_scalar = 0.f;
    for(size_t n = 0; n < lep_p4.size(); ++n) {
        if(lep_genTauIndex.at(n) >= 0) {
            h_tautau_pt_scalar += lep_p4.at(n).Pt();
            n_tau++;
        }
    }
    if(n_tau != 2)
        throw std::runtime_error("too few taus");

    return h_tautau_pt_scalar;
}

ROOT::VecOps::RVec<float> MakeDeepFlavour_bVSall (const ROOT::VecOps::RVec<float>& jets_deepFlavour_b,
                                                  const ROOT::VecOps::RVec<float>& jets_deepFlavour_bb,
                                                  const ROOT::VecOps::RVec<float>& jets_deepFlavour_lepb)
{
    ROOT::VecOps::RVec<float> b_vs_all(jets_deepFlavour_b.size());
    for(size_t n = 0; n < jets_deepFlavour_b.size(); ++n)
        b_vs_all.at(n) = jets_deepFlavour_b.at(n) + jets_deepFlavour_bb.at(n) + jets_deepFlavour_lepb.at(n);
    return b_vs_all;
}

float jets_deepFlavour (const ROOT::VecOps::RVec<float>& deepFlavour_bVSall,
                        const ROOT::VecOps::RVec<size_t>& df_index, const size_t n)
{
    if(n < df_index.size()){
        size_t selected_index = df_index.at(n);
        return deepFlavour_bVSall.at(selected_index);
    }
    else
        return 0;
}

float jets_deepCSV (const ROOT::VecOps::RVec<float>& deepCsv_BvsAll,
                    const ROOT::VecOps::RVec<size_t>& df_index, const size_t n)
{
    if(n < df_index.size()){
        size_t selected_index = df_index.at(n);
        return deepCsv_BvsAll.at(selected_index);
    }
    else
        return 0;
}


float jet_p4_pt (const ROOT::VecOps::RVec<size_t>& df_index,
                 const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4, const size_t n)
{   if(n < df_index.size())
        return jets_p4.at(df_index.at(n)).pt();
    else
        return 0;
}

float jet_p4_eta (const ROOT::VecOps::RVec<size_t>& df_index,
                  const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4, const size_t n)
{   if(n < df_index.size())
        return jets_p4.at(df_index.at(n)).eta();
    else
        return 0;
}

float jet_p4_E (const ROOT::VecOps::RVec<size_t>& df_index,
                const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4, const size_t n)
{   if(n < df_index.size())
        return jets_p4.at(df_index.at(n)).E();
    else
        return 0;
}

float jet_p4_M (const ROOT::VecOps::RVec<size_t>& df_index,
                const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4, const size_t n)
{   if(n < df_index.size())
        return jets_p4.at(df_index.at(n)).M();
    else
        return 0;
}

float rel_jet_M_pt (const ROOT::VecOps::RVec<size_t>& df_index,
                    const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4, const size_t n)
{   if(n < df_index.size())
        return jets_p4.at(df_index.at(n)).M() / jets_p4.at(df_index.at(n)).Pt();
    else
        return 0;
}

float rel_jet_E_pt (const ROOT::VecOps::RVec<size_t>& df_index,
                    const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4, const size_t n)
{   if(n < df_index.size())
        return jets_p4.at(df_index.at(n)).E() / jets_p4.at(df_index.at(n)).Pt();
    else
        return 0;
}


int jet_genbJet (const ROOT::VecOps::RVec<int>& jet_genJetIndex,
                 const ROOT::VecOps::RVec<size_t>& ordered_jet_indexes, const size_t n,
                 const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4)
{
    if(n < ordered_jet_indexes.size()){
        size_t index = ordered_jet_indexes.at(n);
        return jet_genJetIndex.at(index) >= 0;
    }
    else
        return 0;
}

ROOT::VecOps::RVec<int> MakeGenbJet (const ROOT::VecOps::RVec<int>& jet_genJetIndex,
                                     const ROOT::VecOps::RVec<size_t>& ordered_jet_indexes)
{
    ROOT::VecOps::RVec<int> jets_genbJet(ordered_jet_indexes.size());
    for(size_t n = 0; n < ordered_jet_indexes.size(); ++n){
        const size_t index = ordered_jet_indexes.at(n);
        jets_genbJet.at(n) = jet_genJetIndex.at(index) >= 0;
    }
    return jets_genbJet;
}

ROOT::VecOps::RVec<size_t> CreateOrderedIndex (const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4,
                                               const ROOT::VecOps::RVec<float>& jets_deepFlavour, bool apply_acceptance,
                                               const ROOT::VecOps::RVec<int>& tau_indeces,
                                               const ROOT::VecOps::RVec<LorentzVectorM>& lep_p4,
                                               size_t max_jet = std::numeric_limits<size_t>::max())
{
    ROOT::VecOps::RVec<size_t> ordered_index;
    for(size_t jet_index = 0; jet_index < jets_p4.size() && ordered_index.size() < max_jet; ++jet_index) {
        if(apply_acceptance && (jets_p4.at(jet_index).pt() < 20 || abs(jets_p4.at(jet_index).eta()) > 2.4)) continue;
        bool has_overlap_with_taus = false;
        for(size_t lep_index = 0; lep_index < tau_indeces.size() && !has_overlap_with_taus; ++lep_index) {
            if(ROOT::Math::VectorUtil::DeltaR(lep_p4.at(tau_indeces.at(lep_index)), jets_p4.at(jet_index)) < 0.5)
                has_overlap_with_taus = true;
        }
        if(!has_overlap_with_taus)
            ordered_index.push_back(jet_index);
    }

    std::sort(ordered_index.begin(), ordered_index.end(), [&](size_t a, size_t b){
        return jets_deepFlavour.at(a) > jets_deepFlavour.at(b);
    });

    return ordered_index;
}

float httDeltaPhi_jet (const LorentzVectorM& htt_p4,const ROOT::VecOps::RVec<size_t>& df_index,
                       const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4, const size_t n)
{
    if(n < df_index.size())
        return ROOT::Math::VectorUtil::DeltaPhi(htt_p4, jets_p4.at(df_index.at(n)));
    else
        return 0;
}

float httDeltaEta_jet (const LorentzVectorM& htt_p4, const ROOT::VecOps::RVec<size_t>& df_index,
                       const ROOT::VecOps::RVec<LorentzVectorE>& jets_p4, size_t n)
{
    if(n < df_index.size())
        return (htt_p4.eta() - jets_p4.at(df_index.at(n)).eta());
    else
        return 0;
}

inline int ToLegacySampleType(int new_sample_type)
{
    static const std::map<int, int> type_map = {
        { 1, 2 }, {2, 2}, 
        { 3, 3 }, {4, 3}, 
        { 17, 0 }, {31, 1}

    };
    auto iter = type_map.find(new_sample_type);
    if(iter == type_map.end())
        throw std::runtime_error("Unknown sample type");
    return iter->second;
}

inline int ToLegacyChannel(int new_channel)
{
	static const std::map<int, int> channel_map = {
		{13, 0}, //etau
		{23, 1}, //mutau
		{33, 2},//tautau
		{11, -1},
		{22, -1},
        {12, -1}
	};
	auto iter = channel_map.find(new_channel);
	if(iter == channel_map.end())
		throw std::runtime_error("Unknown channel ID");
	return iter->second;
}

inline int ToPairType(int new_channel)
{
    static const std::map<int, int> channel_map = {
        {13, 1}, //etau
        {23, 0}, //mutau
        {33, 2},//tautau
        {11, -1},
        {22, -1},
        {12, -1}
    };
    auto iter = channel_map.find(new_channel);
    if(iter == channel_map.end())
        throw std::runtime_error("Unknown channel ID");
    return iter->second;
}

inline int ToLegacyYear(int new_year)
{
	static const std::map<int, int> year_map = {
		{2, 2016}, 
		{3, 2017},
		{4, 2018},
        {5, 2022},
        {6, 2022},
        {7, 2023},
        {8, 2023}
	};
	auto iter = year_map.find(new_year);
	if(iter == year_map.end())
		throw std::runtime_error("Unknown year");
	return iter->second;
}

inline int ToEraId(int new_eraid)
{
    static const std::map<int, int> era_map = {
        {5, 0},
        {6, 1},
        {7, 2},
        {8, 3}
    };
    auto iter = era_map.find(new_eraid);
    if(iter == era_map.end())
        throw std::runtime_error("Unknown era");
    return iter->second;
}
			

RVecS CreateIndexes(size_t vecSize){
  RVecS i(vecSize);
  std::iota(i.begin(), i.end(), 0);
  return i;
}

template<typename V>
RVecI ReorderObjects(const V& varToOrder, const RVecI& indices, size_t nMax=std::numeric_limits<size_t>::max())
{
  RVecI ordered_indices = indices;
  std::sort(ordered_indices.begin(), ordered_indices.end(), [&](int a, int b) {
    return varToOrder.at(a) > varToOrder.at(b);
  });
  const size_t n = std::min(ordered_indices.size(), nMax);
  ordered_indices.resize(n);
  return ordered_indices;
}
