import copy
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from rich import print


def get_subdirs(path):
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return [] # if path doesn't exist, give empty string
    return [d.name for d in p.iterdir() if d.is_dir()]

def dimuMassPlot(df, save_path):
    # --- Define regions ---
    barrel_cut = 0.9  # threshold for barrel 
    endcap_cut_low = 1.8  # threshold for endcap
    endcap_cut_high = 2.4  # threshold for endcap
    
    barrel_barrel = df[(df["mu1_eta"].abs() < barrel_cut) & (df["mu2_eta"].abs() < barrel_cut)]
    endcap_endcap = df[
        (endcap_cut_low < df["mu1_eta"].abs()) &  
        (df["mu1_eta"].abs()  < endcap_cut_high) & 
        (endcap_cut_low < df["mu2_eta"].abs()) & 
        (df["mu2_eta"].abs() < endcap_cut_high)
         ]
    
    # --- Plot histograms ---
    plt.hist(barrel_barrel["dimuon_ebe_mass_res"]/barrel_barrel["dimuon_mass"], bins=40, alpha=0.6, label="Barrel-Barrel", histtype="step", density=True)
    plt.hist(endcap_endcap["dimuon_ebe_mass_res"]/endcap_endcap["dimuon_mass"], bins=40, alpha=0.6, label="Endcap-Endcap", histtype="step", density=True)

    plt.xlim(0, 0.05)
    plt.xlabel("EBE Dimuon mass resolution / dimuon mass")
    plt.ylabel("A.U.")
    plt.legend()
    plt.title("Dimuon mass resolution in BB and EE regions")
    plt.savefig(f"{save_path}/dimuMassBB_EE.pdf")


def dimuMassResScatterPlot(df, save_path):
    # Scatter plot
    # plt.figure(figsize=(6,4))
    plt.ylim(0, 10)
    plt.scatter(df["mu1_eta"], df["dimuon_ebe_mass_res"], alpha=0.05)
    plt.xlabel("mu1_eta")
    plt.ylabel("dimuon_mass_res")
    plt.title("Scatter plot: dimuon_mass_res vs mu1_eta")
    plt.grid(True)
    plt.savefig(f"{save_path}/dimuResMu1EtaScatter.png")
    plt.clf()


    # mu2 eta
    plt.ylim(0, 10)
    plt.scatter(df["mu2_eta"], df["dimuon_ebe_mass_res"], alpha=0.05)
    plt.xlabel("mu2_eta")
    plt.ylabel("dimuon_mass_res")
    plt.title("Scatter plot: dimuon_mass_res vs mu2_eta")
    plt.savefig(f"{save_path}/dimuResMu2EtaScatter.png")
    plt.clf()
    
    
    # dimuon rap
    plt.ylim(0, 10)
    plt.scatter(df["dimuon_rapidity"], df["dimuon_ebe_mass_res"], alpha=0.05)
    plt.xlabel("dimuon_rapidity")
    plt.ylabel("dimuon_mass_res")
    plt.title("Scatter plot: dimuon_mass_res vs dimuon_rapidity")
    plt.savefig(f"{save_path}/dimuResDimuRapScatter.png")
    plt.clf()

    # bdt wgt
    # plt.ylim(0, 10)
    dataset_l = df["dataset"].unique()
    for dataset in dataset_l:
        subset = df[df["dataset"] == dataset]
        plt.scatter(subset["dimuon_mass"], subset["bdt_wgt"], alpha=0.05, label=dataset)
        plt.legend()
        plt.xlabel("dimuon_mass")
        plt.ylabel("bdt_wgt")
        plt.title("Scatter plot: dimuon_mass_res vs dimuon_rapidity")
        plt.savefig(f"{save_path}/dimuMBdtWgtScatter_{dataset}.png")
        plt.clf()
    
    
def plotBdtWgt(df, save_path):
    # --- Mass window cut ---
    mass_bins_edges = [115, 120, 125, 130, 135]
    dataset_l = df["dataset"].unique()
    for low, high in zip(mass_bins_edges[:-1], mass_bins_edges[1:]):
        mass_mask = (df["dimuon_mass"] > low) & (df["dimuon_mass"] < high)
        df_sel = df[mass_mask]
        # wgt_name = 'wgt_nominal_orig'
        wgt_name = 'bdt_wgt'
        print(f"any neg wgt: {np.any((df['bdt_wgt'] <0))}")
        print(f"any neg wgt wgt_nominal: {np.any((df['wgt_nominal_orig'] <0))}")
        # ===================================================
        # 1) Overlay all datasets in a single plot
        # ===================================================
        # plt.figure(figsize=(6,4))
        
        for dataset in dataset_l:
            subset = df_sel[df_sel["dataset"] == dataset]
            plt.hist(
                subset[wgt_name],
                bins=30,
                histtype="step",
                label=dataset
            )

        plt.xlim(-7e-6, 7e-6)
        
        plt.xlabel(wgt_name)
        plt.ylabel("Counts")
        plt.title(f"wgt_name distribution per dataset\n{low} $ < mMuMu < $ {high}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        # plt.show()
        plt.savefig(f"{save_path}/bdtWgts{low}to{high}.pdf")
        plt.clf()

        # --------------------------------------------------------------
        for dataset in dataset_l:
            subset = df_sel[df_sel["dataset"] == dataset]
            plt.hist(
                subset[wgt_name]/subset["dimuon_ebe_mass_res"],
                bins=30,
                histtype="step",
                label=dataset
            )
        
        plt.xlabel(wgt_name)
        plt.ylabel("Counts")
        plt.title(f"wgt_name distribution per dataset\n{low} $ < mMuMu < $ {high}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        # plt.show()
        plt.savefig(f"{save_path}/bdtWgtsDivDimuonMassRes{low}to{high}.pdf")
        plt.clf()
        
        
        # ===================================================
        # 2) One separate figure per dataset
        # ===================================================
        for dataset in dataset_l:
            subset = df_sel[df_sel["dataset"] == dataset]
            # plt.figure(figsize=(6,4))
            plt.hist(
                subset[wgt_name],
                bins=30,
                histtype="step",
                color="blue",
                label=dataset
            )
            plt.xlabel(wgt_name)
            plt.ylabel("Counts")
            plt.title(f"{wgt_name} distribution for {dataset}\n{low} $ < mMuMu < $ {high}")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            # plt.show()
            plt.savefig(f"{save_path}/{dataset}_bdtWgts{low}to{high}.pdf")
            plt.clf()

def pair_and_remove(df, cols=("mu1_eta","mu2_eta"), wgt_col="wgt_nominal"):
    """
    Pair each negative-weight row with at most one positive-weight row
    using Hungarian algorithm (minimizing L1 distance).
    Remove the paired rows from the original df.

    Returns:
        matches_df: DataFrame with match info (neg_idx, pos_idx, dist, ...)
        remaining_df: df with matched rows removed
    """    
    # Split
    neg = df[df[wgt_col] < 0].copy()
    pos = df[df[wgt_col] > 0].copy()

    if len(neg) == 0 or len(pos) == 0:
        return pd.DataFrame(), df.copy()

    # Arrays
    X = neg.loc[:, cols].to_numpy(dtype=float)
    Y = pos.loc[:, cols].to_numpy(dtype=float)
    # print(f"X: {X}")
    # print(f"y: {Y}")
    # Cost matrix (L1 distance)
    cost = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)
    # print(f"cost: {cost}")
    
    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    # print(f"row_ind: {row_ind}")
    # print(f"col_ind: {col_ind}")
    
    # Map back to indices
    neg_idx = neg.index.to_numpy()[row_ind]
    pos_idx = pos.index.to_numpy()[col_ind]
    dists   = cost[row_ind, col_ind]

    # Matches dataframe
    data = {
        "neg_idx": neg_idx,
        "pos_idx": pos_idx,
        "dist": dists,
        "neg_wgt": df.loc[neg_idx, wgt_col].to_numpy(),
        "pos_wgt": df.loc[pos_idx, wgt_col].to_numpy(),
    }
    for c in cols:
        data[f"neg_{c}"] = df.loc[neg_idx, c].to_numpy()
        data[f"pos_{c}"] = df.loc[pos_idx, c].to_numpy()

    matches_df = pd.DataFrame(data).sort_values("dist").reset_index(drop=True)

    # Drop matched rows from original df
    matched_indices = np.concatenate([neg_idx, pos_idx])
    remaining_df = df.drop(index=matched_indices).reset_index(drop=True)

    return matches_df, remaining_df

# def PairNAnnhilateNegWgt(df):
#     print(f"df len {len(df)}")
    
#     datasets = df["dataset"].unique()
#     # year_param_name = "bdt_year"
#     year_param_name = "year"
#     years = df[year_param_name].unique()
#     # Make an empty copy of df (same columns, no rows)
#     df_out = df.iloc[0:0].copy()
    
#     # Iterate over unique datasets
#     for year in years:
#         for ds in datasets:
#             # Select rows matching this dataset
#             subset = df[(df["dataset"] == ds) & (df[year_param_name] == year)].copy()
#             if np.all(subset["wgt_nominal_orig"] >=0):
#                 print(f"no neg wgts in {ds} {year}")
#             else:
#                 print(f"neg wgts in {ds} {year}")
#                 # --- Preprocess here ---
#                 print(f"subset len b4 {year} {len(subset)}")
                
#                 colsOfInterest = ["mu1_eta", "mu2_eta","dimuon_pt"]
#                 _, subset = pair_and_remove(subset, cols=colsOfInterest, wgt_col="wgt_nominal_orig")
#                 print(f"subset len after {year} {len(subset)}")
    
#                 # # sanity check:
#                 # assert np.all(subset["wgt_nominal_orig"] >=0)
        
#             # Append processed rows into the output df
#             df_out = pd.concat([df_out, subset], ignore_index=True)

#     print(f"final df_out len {len(df_out)}")
#     # print(df_out)
#     # raise ValueError
#     df_out = df_out[df_out["wgt_nominal_orig"] >=0] # FIXME. we see two entries (so very few) that still have negative events, so temp solution. The two entries are from one of the none DY bkg events.
#     return df_out

def split_into_n_parts(lst, n_parts):
    return [lst[i::n_parts] for i in range(n_parts)]
    
def split_df(df: pd.DataFrame, max_num_rows: int):
    """
    Helper function that splits pd df into smaller dfs
    """
    for i in range(0, len(df), max_num_rows):
        yield df.iloc[i : i + max_num_rows]

def PairNAnnhilateNegWgt(df, max_num_rows=80_000):
    print(f"df len {len(df)}")
    print(f"max_num_rows {max_num_rows}")
    datasets = df["dataset"].unique()
    year_param_name = "year"
    years = df[year_param_name].unique()
    
    # Make an empty copy of df (same columns, no rows)
    df_out = df.iloc[0:0].copy()
    
    # Iterate over unique datasets
    for year in years:
        for ds in datasets:
            # Select rows matching this dataset
            subset_big = df[(df["dataset"] == ds) & (df[year_param_name] == year)].copy()
            for subset in split_df(subset_big, max_num_rows):
                if np.all(subset["wgt_nominal_orig"] >=0):
                    print(f"we detect no neg wgts in {ds} yr {year}")
                else:
                    print(f"we detect neg wgts in {ds} yr {year}")
                    # --- Preprocess here ---
                    print(f"subset len b4 {year}: {len(subset)}")
                    
                    colsOfInterest = ["mu1_eta", "mu2_eta","dimuon_pt"]
                    _, subset = pair_and_remove(subset, cols=colsOfInterest, wgt_col="wgt_nominal_orig")
                    print(f"subset len after {year}: {len(subset)}")
        
                    # # sanity check:
                    # assert np.all(subset["wgt_nominal_orig"] >=0)
            
                # Append processed rows into the output df
                df_out = pd.concat([df_out, subset], ignore_index=True)

    print(f"final df_out len {len(df_out)}")
    df_out = df_out[df_out["wgt_nominal_orig"] >=0] # FIXME. we see two entries (so very few) that still have negative events, so temp solution. The two entries are from one of the none DY bkg events.
    return df_out



def PairNAnnhilateNegWgt_inChunks(df, max_num_rows=80_000):
    print(f"max_num_rows: {max_num_rows}")
    processed_chunks = []
    for chunk in split_df(df, max_num_rows):
        processed_chunks.append(PairNAnnhilateNegWgt(chunk, max_num_rows=max_num_rows))
    print(f"PairNAnnhilateNegWgt_inChunks processed_chunks len: {len(processed_chunks)}")
    return pd.concat(processed_chunks, axis=0)  # preserves chunk order

# def fillNanJetvariables(df, forward_filter, jet_variables):
#     dijet_variables = [ 
#         # 'jet1_eta', 
#         # 'jet2_eta', 
#         # 'jet1_pt', 
#         # 'jet2_pt', 
#         'jj_dEta', 
#         'jj_dPhi', 
#         'jj_mass', 
#         'mmj_min_dEta', 
#         'mmj_min_dPhi', 
#     ]
#     jet_variables = list(set(jet_variables + dijet_variables))
#     jet_variables  = [var+"_nominal" for var in jet_variables] 
#     # print(f"df.loc[forward_filter, jet_variables] b4: {df.loc[forward_filter, jet_variables]}")
#     # df.loc[forward_filter, jet_variables].to_csv("dfb4.csv")
#     # print(f"forward_filter: {forward_filter}")
#     # print(f"jet_variables: {jet_variables}")
    
#     for jet_var in jet_variables:
#         if jet_var in df.columns:
#             if "dPhi" in jet_var:
#                 df.loc[forward_filter, jet_var] = -999.0
#             else:
#                 df.loc[forward_filter, jet_var] = -999.0

#     # print(f"df.loc[forward_filter, jet_variables] after: {df.loc[forward_filter, jet_variables]}")
#     # df.loc[forward_filter, jet_variables].to_csv("dfafter.csv")

#     # remove njets
#     df.loc[forward_filter, "njets_nominal"] = df.loc[forward_filter, "njets_nominal"] - 1
#     print(f'np.all(df["njets_nominal"]): {np.all(df["njets_nominal"])}')
#     assert(np.all(df["njets_nominal"]>=0))
#     return df
    
# def removeForwardJets(df):
#     """
#     remove jet variables that are in the forward region abs(jet eta)> 2.5 with with fill nan
#     values consistent with the rest of the framework for ggH BDT.
#     """
#     df_new = copy.deepcopy(df)
    
#     # leading jet  --------------------------
#     forward_filter = (abs(df["jet1_eta_nominal"]) > 2.5) & (abs(df["jet1_eta_nominal"]) < 5.0)
#     jet_variables = [
#         "jet1_eta",
#         "jet1_pt",
#         # "jet1_phi",
#         # "jet1_mass",
#         "mmj1_dEta",
#         # "mmj1_dPhi",
#     ]
#     df_new = fillNanJetvariables(df_new, forward_filter, jet_variables)
#     # raise ValueError
    
#     # sub-leading jet  --------------------------
#     forward_filter = (abs(df["jet2_eta_nominal"]) > 2.5) & (abs(df["jet2_eta_nominal"]) < 5.0)
#     jet_variables = [
#         "jet2_eta",
#         "jet2_pt",
#         # "jet2_phi",
#         # "jet2_mass",
#     ]
#     df_new = fillNanJetvariables(df_new, forward_filter, jet_variables)
#     return df_new

def auc_from_eff(eff_sig, eff_bkg):
    fpr = 1.0 - np.asarray(eff_bkg)
    tpr = np.asarray(eff_sig)
    # sort by FPR ascending before integrating
    order = np.argsort(fpr)

    # in some numpy packages np.trapezoid doesn't exist.
    # then use np.trapz instead
    function_name = 'trapezoid'
    alternative_function = 'trapz'
    
    if hasattr(np, function_name):
        trapz_func = getattr(np, function_name)
    else:
        trapz_func = getattr(np, alternative_function)
        
    return trapz_func(tpr[order], fpr[order])


def addErrByQuadrature(df, columns=[]):
    rel_errByCol = []
    for col in columns:
        val = df[col]
        val_err = df[f"{col}_err"]
        rel_err = val_err/val
        rel_err = np.nan_to_num(rel_err, nan=0.0)
        rel_errByCol.append(rel_err)
    print(f"rel_errByCol: {len(rel_errByCol)}")
    rel_errByCol = np.array(rel_errByCol).T
    print(f"rel_errByCol: {(rel_errByCol)}")
    print(f"rel_errByCol: {(rel_errByCol.shape)}")
    rel_errByQuad = np.sqrt(np.sum(rel_errByCol**2, axis=1))
    print(f"rel_errByQuad len: {(rel_errByQuad.shape)}")
    print(f"rel_errByQuad: {rel_errByQuad}")
    print(f"rel_errByQuad: {np.mean(rel_errByQuad)}")
    df["relErrTotal"] = rel_errByQuad
    return df

def GetAucStdErrHanleyMcNeil(auc, n_positive, n_negative):
    """
    Source: https://jhanley.biostat.mcgill.ca/software/Hanley_McNeil_Radiology_82.pdf
    
    """
    print(f"auc: {auc}")
    print(f"n_positive: {n_positive}")
    print(f"n_negative: {n_negative}")
    Q1 = auc/(2-auc)
    Q2 = 2*auc**2/(1+auc)
    var_auc = auc*(1-auc)
    var_auc += (n_positive-1)*(Q1-auc**2)
    var_auc += (n_negative-1)*(Q2-auc**2)
    var_auc = var_auc / (n_positive*n_negative)
    return np.sqrt(var_auc)
    
# def processYearCol(df):
#     df_new = copy.deepcopy(df)
#     print(df_new["year"].unique())
#     raise ValueError
    
#     # Mapping dictionary
#     replace_map = {
#         "2016preVFP": 2015,
#         "2016postVFP": 2016
#     }
    
#     df_new["year"] = df_new["year"].replace(replace_map)
#     return df_new

def plot6_5FromHist(hist_dict: dict, edges, save_path:str, save_fname: str, unifiedColorScheme=False):
    import mplhep as hep
    plt.style.use(hep.style.CMS)


    # year_colors = {
    #     "2016": "teal",
    #     "2017": "orange",
    #     "2018": "green",
    #     "all": "purple",
    # }
    
    # Year-based shade variations
    year_flavors = ["2016", "2017", "2018", "all"]
    
    # Define color “families” for each year
    bkg_colors = {
        "2016": "#d73027",  # dark red
        "2017": "#fc8d59",  # orange-red
        "2018": "#fdae61",  # peach-red
        # "all":  "#f46d43",  # coral
        "all":  "purple",  # coral
    }
    
    sig_colors = {
        "2016": "#4575b4",  # dark blue
        "2017": "#74add1",  # sky blue
        "2018": "#abd9e9",  # light blue
        # "all":  "#3288bd",  # mid blue
        "all":  "purple",  # mid blue
    }
    
    # plot
    fig, ax_main = plt.subplots(figsize=(10, 8.6))
    for hist_name, hist in hist_dict.items():
        if unifiedColorScheme:
            # color = "red" if "Bkg" in hist_name else "blue"
            # color = next((c for y, c in year_colors.items() if y in hist_name), "gray") # Pick color based on which year substring appears in the hist_name
            # Determine year
            year = next((y for y in year_flavors if y in hist_name), "all")
        
            # Choose color family
            if "Bkg" in hist_name:
                color = bkg_colors[year]
            else:
                color = sig_colors[year]
            ax_main.stairs(hist, edges, label = hist_name, color=color)
        else:
            ax_main.stairs(hist, edges, label = hist_name)
    
    
        
    # Add legend and axis labels
    ax_main.set_xlabel('BDT Score')
    ax_main.set_ylabel("a.u.")
    # ax_main.legend()
    # --- Sort legend alphabetically ---
    handles, labels = ax_main.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))  # sort by label text
    ax_main.legend(handles, labels, ncol=2)

    # Set Range
    # ax_main.set_xlim(-0.9, 0.9)
    ax_main.set_xlim(-1.0, 1.0)
    ax_main.set_xticks([ -0.8, -0.6, -0.4, -0.2 , 0. ,  0.2 , 0.4 , 0.6,  0.8])
    ax_main.set_ylim(0, 0.09)
    
    # hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, lumi=lumi, ax=ax_main)
    hep.cms.label(data=False, ax=ax_main)
    
    plt.savefig(f"{save_path}/{save_fname}.pdf")
    plt.clf()


def customROC_curve_AN(label, pred, weight, doClassBalance = False):
    """
    generates signal and background efficiency consistent with the AN,
    as described by Fig 4.6 of Dmitry's PhD thesis
    """
    # we assume sigmoid output with labels 0 = background, 1 = signal
    thresholds = np.linspace(start=0,stop=1, num=500) 
    effBkg_total = -99*np.ones_like(thresholds) # effBkg = false positive rate
    effSig_total = -99*np.ones_like(thresholds) # effSig = true positive rate
    FP_total = -99*np.ones_like(thresholds) 
    TP_total = -99*np.ones_like(thresholds) 
    TN_total = -99*np.ones_like(thresholds) 
    FN_total = -99*np.ones_like(thresholds) 
    FP_err_total = -99*np.ones_like(thresholds) 
    TP_err_total = -99*np.ones_like(thresholds) 
    TN_err_total = -99*np.ones_like(thresholds) 
    FN_err_total = -99*np.ones_like(thresholds) 

    # apply class balance if requested
    if doClassBalance:
        weight = copy.deepcopy(weight)
        is_bkg = label == 0
        is_sig = ~is_bkg
        bkg_weight = weight[is_bkg]
        weight = np.where(is_bkg, weight/np.sum(bkg_weight), weight)
        sig_weight = weight[is_sig]
        # weight = np.where(is_sig, weight/np.sum(sig_weight), weight)
        weight = np.where(is_sig, np.ones_like(weight), weight)
        print(f"bkg wgt sum: {np.sum(weight[is_bkg])}")
        print(f"sig wgt sum: {np.sum(weight[is_sig])}")
        # weight = np.ones_like(weight)
    for ix in range(len(thresholds)):
        threshold = thresholds[ix]
        # get FP and TP
        positive_filter = (pred > threshold)
        falsePositive_filter = positive_filter & (label == 0)
        FP = np.sum(weight[falsePositive_filter])#  FP = false positive
        truePositive_filter = positive_filter & (label == 1)
        TP = np.sum(weight[truePositive_filter])#  TP = true positive
        

        # get TN and FN
        negative_filter = (pred <= threshold) # just picked negative to be <=
        trueNegative_filter = negative_filter & (label == 0)
        TN = np.sum(weight[trueNegative_filter])#  TN = true negative
        falseNegative_filter = negative_filter & (label == 1)
        FN = np.sum(weight[falseNegative_filter])#  FN = false negative

        # obtain the err of FP, TP, TN, FN
        FP_err = np.sqrt(np.sum(weight[falsePositive_filter]**2))
        TP_err = np.sqrt(np.sum(weight[truePositive_filter]**2))
        TN_err = np.sqrt(np.sum(weight[trueNegative_filter]**2))
        FN_err = np.sqrt(np.sum(weight[falseNegative_filter]**2))
        


        # effBkg = TN / (TN + FP) # Dmitry PhD thesis definition
        # effSig = FN / (FN + TP) # Dmitry PhD thesis definition
        effBkg = FP / (TN + FP) # AN-19-124 ggH Cat definition
        effSig = TP / (FN + TP) # AN-19-124 ggH Cat definition
        effBkg_total[ix] = effBkg
        effSig_total[ix] = effSig

        # print(f"ix: {ix}") 
        # print(f"threshold: {threshold}")
        # print(f"effBkg: {effBkg}")
        # print(f"effSig: {effSig}")

        FP_total[ix] = FP
        TP_total[ix] = TP
        TN_total[ix] = TN
        FN_total[ix] = FN
        FP_err_total[ix] = FP_err
        TP_err_total[ix] = TP_err
        TN_err_total[ix] = TN_err
        FN_err_total[ix] = FN_err
        
        
        # sanity check
        assert ((np.sum(positive_filter) + np.sum(negative_filter)) == len(pred))
        total_yield = FP + TP + FN + TN
        assert(np.isclose(total_yield, np.sum(weight)))
        # print(f"total_yield: {total_yield}")
        # print(f"np.sum(weight): {np.sum(weight)}")
    
    # print(f"np.sum(effBkg_total ==-99) : {np.sum(effBkg_total ==-99)}")
    # print(f"np.sum(effSig_total ==-99) : {np.sum(effSig_total ==-99)}")
    # neither_zeroNorOne = ~((label == 0) | (label == 1))
    # print(f"np.sum(neither_zeroNorOne) : {np.sum(neither_zeroNorOne)}")
    effBkg_total[np.isnan(effBkg_total)] = 1
    effSig_total[np.isnan(effSig_total)] = 1
    # print(f"effBkg_total: {effBkg_total}")
    # print(f"effSig_total: {effSig_total}")
    # print(f"thresholds: {thresholds}")
    # raise ValueError
    effBkgSig_df = pd.DataFrame({
        "FP" : FP_total,
        "TP" : TP_total,
        "TN" : TN_total,
        "FN" : FN_total,
        "FP_err" : FP_err_total,
        "TP_err" : TP_err_total,
        "TN_err" : TN_err_total,
        "FN_err" : FN_err_total,
    })
    
    return (effBkg_total, effSig_total, thresholds, effBkgSig_df)

def fullROC_operations(fig, data_dict, save_path, year, label, doClassBalance=False):
    if doClassBalance:
        # save_str_addendum = "_clsWgtBal"
        # save_str_addendum = "_noWgt"
        save_str_addendum = "_eqlSigWgt"
    else:
        save_str_addendum = ""
    y_train = data_dict["y_train"]
    y_pred_train = data_dict["y_pred_train"]
    weight_nom_train = data_dict["weight_nom_train"]

    y_val = data_dict["y_val"]
    y_pred = data_dict["y_pred"]
    weight_nom_val = data_dict["weight_nom_val"]

    y_eval = data_dict["y_eval"]
    y_eval_pred = data_dict["y_eval_pred"]
    weight_nom_eval = data_dict["weight_nom_eval"]
    
    
    
    eff_bkg_train, eff_sig_train, thresholds_train, TpFpTnFn_df_train = customROC_curve_AN(y_train, y_pred_train, weight_nom_train, doClassBalance=doClassBalance)
    eff_bkg_val, eff_sig_val, thresholds_val, TpFpTnFn_df_val = customROC_curve_AN(y_val, y_pred, weight_nom_val, doClassBalance=doClassBalance)
    eff_bkg_eval, eff_sig_eval, thresholds_eval, TpFpTnFn_df_eval = customROC_curve_AN(y_eval, y_eval_pred, weight_nom_eval, doClassBalance=doClassBalance)

    # deprecated -----------------------
    # cols2merge = ["TP","FP","TN","FN"]
    # TpFpTnFn_df_train = addErrByQuadrature(TpFpTnFn_df_train, columns=cols2merge)
    # TpFpTnFn_df_val = addErrByQuadrature(TpFpTnFn_df_val, columns=cols2merge)
    # TpFpTnFn_df_eval = addErrByQuadrature(TpFpTnFn_df_eval, columns=cols2merge)
    # deprecated -----------------------
    

    # -------------------------------------
    # save ROC curve
    # -------------------------------------
    # csv_savepath = f"output/bdt_{name}_{year}/rocEffs_{label}{save_str_addendum}.csv"
    csv_savepath = f"{save_path}/rocEffs_{label}{save_str_addendum}.csv"
    
    roc_df = pd.DataFrame({
        "eff_sig_eval" : eff_sig_eval,
        "eff_bkg_eval" : eff_bkg_eval,
        "eff_sig_train" : eff_sig_train,
        "eff_bkg_train" : eff_bkg_train,
        "eff_sig_val" : eff_sig_val,
        "eff_bkg_val" : eff_bkg_val,
    })
    # roc_df = pd.concat([roc_df, TpFpTnFn_df_train, TpFpTnFn_df_val, TpFpTnFn_df_eval], axis=1)
    roc_df.to_csv(csv_savepath)

    
    auc_eval  = auc_from_eff(eff_sig_eval,  eff_bkg_eval)
    auc_train = auc_from_eff(eff_sig_train, eff_bkg_train)
    auc_val   = auc_from_eff(eff_sig_val,   eff_bkg_val)

    # Calculate auc err using HM method
    n_pos_eval = np.sum(y_eval.ravel() ==1)
    n_neg_eval = np.sum(y_eval.ravel() !=1)
    auc_err_eval = GetAucStdErrHanleyMcNeil(auc_eval, n_pos_eval, n_neg_eval)
    n_pos_train = np.sum(y_train.ravel() ==1)
    n_neg_train = np.sum(y_train.ravel() !=1)
    auc_err_train = GetAucStdErrHanleyMcNeil(auc_train, n_pos_train, n_neg_train)
    n_pos_val = np.sum(y_val.ravel() ==1)
    n_neg_val = np.sum(y_val.ravel() !=1)
    auc_err_val = GetAucStdErrHanleyMcNeil(auc_val, n_pos_val, n_neg_val)
    
    print(f"auc_err_train: {auc_err_train}")

    # -------------------------------------
    # save auc and auc err
    # -------------------------------------
    # csv_savepath = f"output/bdt_{name}_{year}/aucInfo_{label}{save_str_addendum}.csv"
    csv_savepath = f"{save_path}/aucInfo_{label}{save_str_addendum}.csv"
    auc_df = pd.DataFrame({
        "auc_eval" : [auc_eval],
        "auc_err_eval" : [auc_err_eval],
        "auc_train" : [auc_train],
        "auc_err_train" : [auc_err_train],
        "auc_val" : [auc_val],
        "auc_err_val" : [auc_err_val],
    })
    # roc_df = pd.concat([roc_df, TpFpTnFn_df_train, TpFpTnFn_df_val, TpFpTnFn_df_eval], axis=1)
    auc_df.to_csv(csv_savepath)

    
    plt.plot(eff_sig_eval, eff_bkg_eval, label=f"ROC (Eval)  — AUC={auc_eval:.4f}+/-{auc_err_eval:.4f}")
    plt.plot(eff_sig_val, eff_bkg_val, label=f"ROC (Val)   — AUC={auc_val:.4f}+/-{auc_err_val:.4f}")
    
    # plt.vlines(eff_sig, 0, eff_bkg, linestyle="dashed")
    plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
    plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
    # plt.hlines(eff_bkg, 0, eff_sig, linestyle="dashed")
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    plt.xlabel('Signal eff')
    plt.ylabel('Background eff')
    plt.yscale("log")
    plt.ylim([0.0001, 1.0])
    
    plt.legend(loc="lower right")
    plt.title(f'ROC curve for ggH BDT {year}')
    # fig.savefig(f"output/bdt_{name}_{year}/log_auc_{label}{save_str_addendum}.pdf")
    fig.savefig(f"{save_path}/log_auc_{label}{save_str_addendum}.pdf")

    
    plt.plot(eff_sig_train, eff_bkg_train, label=f"ROC (Train) — AUC={auc_train:.4f}+/-{auc_err_train:.4f}")
    plt.legend(loc="lower right")
    # fig.savefig(f"output/bdt_{name}_{year}/log_auc_{label}_w_train{save_str_addendum}.pdf")
    fig.savefig(f"{save_path}/log_auc_{label}_w_train{save_str_addendum}.pdf")
    
    plt.clf()
    # superimposed flipped log ROC start --------------------------------------------------------------------------
    plt.plot(1-eff_sig_eval,  1-eff_bkg_eval,  label=f"Stage2 ROC (Eval)  — AUC={auc_eval:.4f}+/-{auc_err_eval:.4f}")
    plt.plot(1-eff_sig_val,   1-eff_bkg_val,   label=f"Stage2 ROC (Val)   — AUC={auc_val:.4f}+/-{auc_err_val:.4f}")

    
    # plt.vlines(eff_sig, 0, eff_bkg, linestyle="dashed")
    plt.vlines(np.linspace(0,1,11), 0, 1, linestyle="dashed", color="grey")
    plt.hlines(np.logspace(-4,0,5), 0, 1, linestyle="dashed", color="grey")
    # plt.hlines(eff_bkg, 0, eff_sig, linestyle="dashed")
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Signal eff')
    plt.ylabel('1- Background eff')
    plt.yscale("log")
    plt.ylim([0.0001, 1.0])
    
    plt.legend(loc="lower right")
    plt.title(f'ROC curve for ggH BDT {year}')
    # fig.savefig(f"output/bdt_{name}_{year}/logFlip_auc_{label}{save_str_addendum}.pdf")
    fig.savefig(f"{save_path}/logFlip_auc_{label}{save_str_addendum}.pdf")

    plt.plot(1-eff_sig_train, 1-eff_bkg_train, label=f"Stage2 ROC (Train) — AUC={auc_train:.4f}+/-{auc_err_train:.4f}")
    plt.legend(loc="lower right")
    # fig.savefig(f"output/bdt_{name}_{year}/logFlip_auc_{label}_w_train{save_str_addendum}.pdf")
    fig.savefig(f"{save_path}/logFlip_auc_{label}_w_train{save_str_addendum}.pdf")
    
    plt.clf()
    return auc_df


def has_bad_values(arr):
    arr = np.asarray(arr, dtype=float)
    return not np.isfinite(arr).all()


def reweightMassToFlat(df, sig_datasets, validation_plot_path, nbins=80, mmin=115, mmax=135, wgt_field="wgt_nominal"):

    # -----------------------------
    # 1) obtain bkg_df
    # -----------------------------
    sig_filter = df["dataset"].isin(sig_datasets)
    sig_df = df[sig_filter] # you leave this alone
    bkg_df = df[~sig_filter] # work on this
    
    # -----------------------------
    # 2) Compute flat reweighting
    # -----------------------------
    edges = np.linspace(mmin, mmax, nbins + 1)
    
    # weighted histogram using original weights
    counts, _ = np.histogram(bkg_df["dimuon_mass"], bins=edges, weights=bkg_df[wgt_field])
    
    # target height per bin (preserve total normalization)
    target = counts.sum() / nbins
    
    eps = 1e-12
    scale = np.where(counts > eps, target / counts, 0.0)
    # assign each event to a bin
    bin_idx = np.digitize(bkg_df["dimuon_mass"], edges) - 1
    in_range = (bin_idx >= 0) & (bin_idx < nbins)
    
    # new flattened weights
    bkg_df["wgt_flat"] = bkg_df[wgt_field]
    bkg_df.loc[in_range, "wgt_flat"] = bkg_df.loc[in_range, wgt_field] * scale[bin_idx[in_range]]
    
    # -----------------------------
    # 3) Histogram before vs after
    # -----------------------------
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    h_old, _ = np.histogram(bkg_df["dimuon_mass"], bins=edges, weights=bkg_df[wgt_field])
    h_new, _ = np.histogram(bkg_df["dimuon_mass"], bins=edges, weights=bkg_df["wgt_flat"])

    print(f"np.sum(h_old): {np.sum(h_old)}")
    print(f"np.sum(h_new): {np.sum(h_new)}")
    plt.figure(figsize=(8,5))
    plt.step(centers, h_old, where="mid", label="Before (exp-shaped)")
    plt.step(centers, h_new, where="mid", label="After (flattened)")
    plt.xlabel("dimuon_mass")
    plt.ylabel("Weighted events / bin")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{validation_plot_path}/bkgMassFlattenRewgtPlot.pdf")
    # -----------------------------
    # 4) Flatness sanity check
    # -----------------------------
    print("Flatness (std/mean) before:", np.std(h_old) / np.mean(h_old))
    print("Flatness (std/mean) after: ", np.std(h_new) / np.mean(h_new))

    # -----------------------------
    # 5) combine the two df back
    # -----------------------------
    sig_df["wgt_flat"] = sig_df[wgt_field]
    df = pd.concat([
        sig_df,
        bkg_df,
    ]).sort_index()
    # overwrite the wgt_field
    bkg_df[wgt_field] = bkg_df["wgt_flat"]
    return df




def apply_gghChannelSelection(delayed_dak_zip):
    """
    small wrapper that takes delayed dask awkward zip and converts them to pandas dataframe
    with zip's keys as columns with extra column "dataset" to be named the string value given
    Note: we also filter out regions not in h-peak region first, and apply cuts for
    ggH production mode
    """
    # filter out arrays not in h_peak
    train_region = (delayed_dak_zip.dimuon_mass > 115.0) & (delayed_dak_zip.dimuon_mass < 135.0) # line 1169 of the AN: when training, we apply a tigher cut
    train_region = ak.fill_none(train_region, value=False)
    # print(f"is_vbf: {is_vbf}")
    # print(f"convert2df train_region:{train_region}")
    # not entirely sure if this is what we use for ROC curve, however

    vbf_cut = ak.fill_none(delayed_dak_zip.jj_mass_nominal > 400, value=False) & ak.fill_none(delayed_dak_zip.jj_dEta_nominal > 2.5, value=False) # for ggH $ VBF
    jet1_cut =  ak.fill_none((delayed_dak_zip.jet1_pt_nominal > 35), value=False) 
    

    prod_cat_cut =  ~(vbf_cut & jet1_cut)
    print("ggH cat!")

    # btag_cut = ak.fill_none((delayed_dak_zip.nBtagLoose_nominal >= 2), value=False) | ak.fill_none((delayed_dak_zip.nBtagMedium_nominal >= 1), value=False)
    btagLoose_filter = ak.fill_none((delayed_dak_zip.nBtagLoose_nominal >= 2), value=False)
    btagMedium_filter = ak.fill_none((delayed_dak_zip.nBtagMedium_nominal >= 1), value=False) & ak.fill_none((delayed_dak_zip.njets_nominal >= 2), value=False)
    btag_cut = btagLoose_filter | btagMedium_filter
   
    category_selection = (
        prod_cat_cut & 
        train_region &
        ~btag_cut # btag cut is for VH and ttH categories
    )
    # print(f"category_selection sum: {ak.sum(category_selection)}")
    computed_zip = delayed_dak_zip[category_selection].compute()
    return computed_zip



def reweightMassToTargetDist_workflow(df, sig_datasets, validation_plot_path, nbins=80, mmin=115, mmax=135, wgt_field="wgt_nominal", target_mass_centre = 91, target_dist_load_path=None):
    """
    wrapper of reweightMassToTargetDist over sig and bkg samples
    """

    # -----------------------------
    # 1) obtain bkg_df
    # -----------------------------
    sig_filter = df["dataset"].isin(sig_datasets)
    sig_df = df[sig_filter] # you leave this alone
    bkg_df = df[~sig_filter] # work on this

    if target_dist_load_path is None:
        target_dist_load_path = os.environ.get("GGH_BDT_TARGET_DIST_PATH")
    if not target_dist_load_path:
        raise FileNotFoundError(
            "Mass decorrelation with a target distribution requires --target_dist_path "
            "or the GGH_BDT_TARGET_DIST_PATH environment variable."
        )
    if not Path(target_dist_load_path).exists():
        raise FileNotFoundError(
            f"Target mass distribution parquet not found: {target_dist_load_path}"
        )
    sig_df = reweightMassToTargetDist(sig_df, target_dist_load_path, mmin, mmax, nbins, validation_plot_path, plot_name="sigMassTargetReWgt", wgt_field=wgt_field, target_mass_centre=target_mass_centre)
    bkg_df = reweightMassToTargetDist(bkg_df, target_dist_load_path, mmin, mmax, nbins, validation_plot_path, plot_name="bkgMassTargetReWgt", wgt_field=wgt_field, target_mass_centre=target_mass_centre)

    # -----------------------------
    # 5) combine the two df back
    # -----------------------------
    df = pd.concat([
        sig_df,
        bkg_df,
    ]).sort_index()
    return df

def recenter_range(x_min, x_max, new_x_center):
    """
    Recenter a range [x_min, x_max] to a new center,
    keeping the same total length.

    Returns:
        new_x_min, new_x_max
    """
    length = x_max - x_min
    half_length = length / 2.0
    
    new_x_min = new_x_center - half_length
    new_x_max = new_x_center + half_length
    
    return new_x_min, new_x_max


def sin_histogram(nbins, xmin, xmax):
    # bin edges
    edges = np.linspace(xmin, xmax, nbins + 1)

    # bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])

    # histogram values following sin(x)
    hist = 1.5 + np.sin(centers)
    return hist

def reweightMassToTargetDist(df, target_dist_load_path, train_x_min, train_x_max, nbins, plot_save_path, plot_name="test", wgt_field="wgt_nominal", target_mass_centre = 91):

    events_target = ak.from_parquet(target_dist_load_path)
    target = ak.to_numpy(events_target.dimuon_mass)
    parquet_wgt_field = "wgt_nominal"
    target_wgt = ak.to_numpy(events_target[parquet_wgt_field])
    
    # Source distribution (to be reweighted)
    # x = np.random.normal(loc=91, scale=6.4, size=50000)
    # w = np.ones_like(x)
    x = df["dimuon_mass"]
    w = df[wgt_field]
    
    # -----------------------------
    # 2. Define binning
    # -----------------------------
    # train_x_min = 115
    # train_x_max = 135
    
    # target_mass_centre = 91
    # -----------------------------
    # Define reference disctribution
    # as histogram 'target_hist'
    # -----------------------------
    if target_mass_centre == "flat": # just get a flat distribution
        print("Taking flat distribution as the Target Hist!")
        target_hist = np.ones(nbins)
    elif target_mass_centre == "sinusoidal":
        nbins = 50
        # xmin, xmax = 0, np.pi
        xmin, xmax = 0, 2*np.pi
        target_hist = sin_histogram(nbins, xmin, xmax)

    else: # else, it's a mass value
        range_ = recenter_range(train_x_min, train_x_max, target_mass_centre)
        target_hist, bin_edges = np.histogram(target, bins=nbins, range=range_, weights=target_wgt)
    
    bin_edges = np.linspace(train_x_min, train_x_max, num=(nbins+1))
    source_hist, _ = np.histogram(x, bins=bin_edges, weights=w)

    print(f"sum source_hist: {np.sum(source_hist)}")
    print(f"sum target_hist b4: {np.sum(target_hist)}")
    
    target_hist = target_hist * (np.sum(source_hist)/np.sum(target_hist)) # match target_hist sum to source_hist sum
    print(f"sum target_hist after: {np.sum(target_hist)}")
    # -----------------------------
    # 3. Compute scale factors
    # -----------------------------
    scale = np.divide(
        target_hist,
        source_hist,
        out=np.zeros_like(target_hist, dtype=float),
        where=source_hist != 0
    )
    
    # -----------------------------
    # 4. Apply per-bin scale factor
    # -----------------------------
    bin_index = np.digitize(x, bin_edges) - 1
    
    # keep only valid bins
    valid = (bin_index >= 0) & (bin_index < len(scale))
    
    w_new = w.copy()
    w_new[valid] *= scale[bin_index[valid]]
    
    # -----------------------------
    # 5. Plot comparison
    # -----------------------------
    train_x_center = (train_x_max + train_x_min)/2
    # if target_mass_centre == "flat":
    if type(target_mass_centre) == str:
        # plt.hist(target, bins=bin_edges, weights=target_wgt, histtype="step", density=True, label="Target")
        print("skip plot flat distribution") # FIXME
    else:
        plot_range_delta = train_x_center - target_mass_centre
        plt.hist(target+plot_range_delta, bins=bin_edges, weights=target_wgt, histtype="step", density=True, label="Target")
        
    plt.hist(x, bins=bin_edges, weights=w, histtype="step", density=True, label="Source (before)")
    
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Histogram Reweighting Example")
    plt.savefig(f"{plot_save_path}/{plot_name}_b4.pdf")
    
    
    plt.hist(x, bins=bin_edges, weights=w_new, histtype="step", density=True, label="Source (after)")
    plt.legend()
    plt.savefig(f"{plot_save_path}/{plot_name}_after.pdf")
    plt.clf()
    
    # -----------------------------
    # 6. re-wgt and return df
    # -----------------------------
    df[wgt_field] =  w_new
    return df
