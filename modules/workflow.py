import copy
import json
import os
import pickle

import awkward as ak
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import auc, roc_curve
import time
from xgboost import XGBClassifier, plot_tree

from modules.utils import fullROC_operations, has_bad_values
from modules.git_utils import get_git_commit, get_git_state
from modules.variables import training_samples

plt.style.use(hep.style.CMS)

def get_xgb_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

# ---- helpers (small + self-contained) ---------------------------------
def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _array_stats(x, name):
    x = np.asarray(x)
    out = {
        "name": name,
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "size": int(x.size),
        "finite_frac": float(np.isfinite(x).mean()) if x.size else None,
    }
    if x.size:
        xf = x[np.isfinite(x)]
        out.update({
            "min": float(np.min(xf)) if xf.size else None,
            "max": float(np.max(xf)) if xf.size else None,
            "mean": float(np.mean(xf)) if xf.size else None,
            "std": float(np.std(xf)) if xf.size else None,
            "sum": float(np.sum(xf)) if xf.size else None,
            "n_nonzero": int(np.count_nonzero(xf)) if xf.size else 0,
        })
    else:
        out.update({"min": None, "max": None, "mean": None, "std": None, "sum": None, "n_nonzero": 0})
    return out

def _df_class_breakdown(df, class_col="class", dataset_col="dataset", wcols=("wgt_nominal_orig", "bdt_wgt")):
    rows = []
    if df is None or df.empty:
        return {"by_dataset_class": [], "overall": {"n": 0}}

    for (ds, cls), g in df.groupby([dataset_col, class_col], dropna=False):
        rec = {"dataset": str(ds), "class": int(cls), "n": int(len(g))}
        for wc in wcols:
            if wc in g.columns:
                w = g[wc].to_numpy()
                rec[f"{wc}_sum"] = float(np.sum(w))
                rec[f"{wc}_sum_abs"] = float(np.sum(np.abs(w)))
                rec[f"{wc}_mean"] = float(np.mean(w)) if len(w) else None
        rows.append(rec)

    overall = {"n": int(len(df))}
    for wc in wcols:
        if wc in df.columns:
            w = df[wc].to_numpy()
            overall[f"{wc}_sum"] = float(np.sum(w))
            overall[f"{wc}_sum_abs"] = float(np.sum(np.abs(w)))
            overall[f"{wc}_min"] = float(np.min(w)) if len(w) else None
            overall[f"{wc}_max"] = float(np.max(w)) if len(w) else None

    return {"by_dataset_class": rows, "overall": overall}

# Convert AUC DataFrames to json-serializable dicts and store in metadata
def _df_to_dict_safe(df):
    if df is None:
        return None
    try:
        return df.to_dict(orient="list")
    except Exception:
        return str(df)

def _write_metadata_json(meta_path, meta_obj):
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta_obj, f, indent=2, sort_keys=True)
# ----------------------------------------------------------------------


def getGOF_KS_bdt(valid_hist, train_hist, weight_val, bin_edges, save_path:str, fold_idx):
    """
    Get KS value for specific value
    """
    # print(f"valid_hist: {valid_hist}")
    # print(f"train_hist: {train_hist}")
    print(f"valid_hist: {np.sum(valid_hist)}")
    print(f"train_hist: {np.sum(train_hist)}")
    
    data_counts = valid_hist
    pdf_counts = train_hist


    plot_line_width = 0.5

    # -------------------------------------------
    # Do KS test
    # -------------------------------------------


    
    data_cdf = np.cumsum(data_counts) / np.sum(data_counts)
    pdf_cdf = np.cumsum(pdf_counts) / np.sum(pdf_counts)
    ks_statistic = np.max(np.abs(data_cdf - pdf_cdf))
    print(f"ks_statistic: {ks_statistic}")
    nevents = weight_val.size
    print(f"nevents: {nevents}")
    
    
    alpha1 = 0.1
    pass_threshold1 = 1.22385 / (nevents**(0.5))
    alpha2 = 0.001
    pass_threshold2 = 1.94947 / (nevents**(0.5))

    df_dict= {
        "ks_statistic": [ks_statistic],
        "nevents" : [nevents],
        "alpha1":[ alpha1],
        "alpha1 pass threshold": [pass_threshold1],
        "alpha1 test pass": [ks_statistic<pass_threshold1],
        "alpha2":[ alpha2],
        "alpha2 pass threshold": [pass_threshold2],
        "alpha2 test pass": [ks_statistic<pass_threshold2],
        
   }
    gof_df = pd.DataFrame(df_dict)
    gof_df.to_csv(f"{save_path}/KS_stats_{fold_idx}.csv")
    

    # Draw the cdf histogram
    
    # bin_centers = np.array(bin_centers)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, data_cdf, label='Validation CDF', linewidth=plot_line_width)
    plt.plot(bin_centers, pdf_cdf, label='Train CDF', linewidth=plot_line_width)
    plt.xlabel('BDT SCORE')
    plt.ylabel('')
    plt.title('BDT distribution of signal sample')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/GoF_cdfs_{fold_idx}.pdf")
    plt.clf()

    # Draw the normalized pdf histogram:
    plt.plot(bin_centers, data_counts/np.sum(data_counts), label='Validation PDF', linewidth=plot_line_width)
    plt.plot(bin_centers, pdf_counts/np.sum(pdf_counts), label='Train PDF', linewidth=plot_line_width)
    plt.xlabel('BDT SCORE')
    plt.ylabel('')
    plt.title('BDT distribution of signal sample')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/GoF_pdfs_{fold_idx}.pdf")
    plt.clf()
        

# def auc_from_eff(eff_sig, eff_bkg):
#     fpr = 1.0 - np.asarray(eff_bkg)
#     tpr = np.asarray(eff_sig)
#     # sort by FPR ascending before integrating
#     order = np.argsort(fpr)
#     return np.trapezoid(tpr[order], fpr[order])


def prepare_features(events, features, variation="nominal"):
    plt.style.use(hep.style.CMS)
    features_var = []
    for trf in features:
        if "soft" in trf:
            variation_current = "nominal"
        else:
            variation_current = variation
        
        if f"{trf}_{variation_current}" in events.fields:
            features_var.append(f"{trf}_{variation_current}")
        elif trf in events.fields:
            features_var.append(trf)
        else:
            print(f"Variable {trf} not found in training dataframe!")
    return features_var

def deactivateWgts(wgt_arr, events, wgts2deactivate):
    """
    """
    wgt_arr_out = copy.deepcopy(wgt_arr) # make a copy bc you never know
    for wgt in wgts2deactivate:
        wgt_arr_out = wgt_arr_out/events[wgt]
    return wgt_arr_out

def generate_normalized_histogram(values, weights, bins):
    """
    Generates a normalized histogram using values and weights.

    Parameters:
    - values: array-like, the data values.
    - weights: array-like, the weights for each data point.
    - bins: int or array-like, defines the bin edges.

    Returns:
    - hist: array, the values of the normalized histogram.
    - bin_edges: array, the edges of the bins.
    """
    # Compute the histogram with weights and density normalization
    hist, bin_edges = np.histogram(values, bins=bins, weights=weights, density=True)
    return hist, bin_edges


def transformBdtOut(pred):
    """
    change the BDT output range from [0,1] to [-1,1]
    """
    pred = pred *2 -1
    return pred

def get6_5(label, pred, weight, save_path:str, name: str):
    # seperate signal and background
    with open("my_data.pkl", "wb") as f:
        pickle.dump((label, pred, weight), f)  #
        
    # binning = np.linspace(start=0,stop=1, num=60) 
    binning = np.linspace(start=-1,stop=1, num=41) 
    bkg_filter = (label ==0)
    bkg_pred = transformBdtOut(pred[bkg_filter])
    bkg_wgt = weight[bkg_filter]
    bkg_wgt = bkg_wgt / np.sum(bkg_wgt) # normalize
    bkg_hist, edges = np.histogram(bkg_pred, bins=binning, weights=bkg_wgt)
    sig_filter = (label ==1)
    sig_pred = transformBdtOut(pred[sig_filter])
    sig_wgt = weight[sig_filter]
    sig_wgt = sig_wgt / np.sum(sig_wgt) # normalize
    sig_hist, _ = np.histogram(sig_pred, bins=binning, weights=sig_wgt)
    # plot
    fig, ax_main = plt.subplots(figsize=(10, 8.6))
    ax_main.stairs(bkg_hist, edges, label = "background", color="Red")
    ax_main.stairs(sig_hist, edges, label = "signal", color="Blue")
    
    
        
    # Add legend and axis labels
    ax_main.set_xlabel('BDT Score')
    ax_main.set_ylabel("a.u.")
    ax_main.legend()
    
    # Set Range
    # ax_main.set_xlim(-0.9, 0.9)
    ax_main.set_xlim(-1.0, 1.0)
    ax_main.set_xticks([ -0.8, -0.6, -0.4, -0.2 , 0. ,  0.2 , 0.4 , 0.6,  0.8])
    ax_main.set_ylim(0, 0.09)
    
    # hep.cms.label(data=True, loc=0, label=status, com=CenterOfMass, lumi=lumi, ax=ax_main)
    hep.cms.label(data=False, ax=ax_main)
    
    plt.savefig(f"{save_path}/6_5_{name}.png")
    plt.savefig(f"{save_path}/6_5_{name}.pdf")
    plt.clf()

    # save hist in pd df
    df_6p5 = pd.DataFrame({
        'bin_left': binning[:-1],
        'bin_right': binning[1:],
        'bin_center': 0.5 * (binning[:-1] + binning[1:]),
        "bkg_hist" : bkg_hist,
        "sig_hist" : sig_hist,
    })
    df_6p5.to_csv(f"{save_path}/6_5_{name}.csv")
    



def convert2df(dak_zip, dataset: str):
    """
    small wrapper that takes delayed dask awkward zip and converts them to pandas dataframe
    with zip's keys as columns with extra column "dataset" to be named the string value given

    Fill missing values; use -1 for dPhi variables (per AN), -999 otherwise.
    """
    nan_val = -999.0
    dphi_nan_val = -1.0

    computed_dict = {}
    for field in dak_zip.fields:
        fill_val = dphi_nan_val if "dPhi" in field else nan_val
        computed_dict[field] = ak.fill_none(dak_zip[field], value=fill_val)

    # Force to numpy arrays (avoid object dtype)
    for k, v in computed_dict.items():
        computed_dict[k] = ak.to_numpy(v)

    df = pd.DataFrame(computed_dict)

    # Replace infs then fill again
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill dPhi NaNs with -1, everything else with -999
    dphi_cols = [c for c in df.columns if "dPhi" in c]
    df.fillna({c: dphi_nan_val for c in dphi_cols}, inplace=True)
    df.fillna(nan_val, inplace=True)

    df["dataset"] = dataset
    df["cls_avg_wgt"] = -1.0

    df["wgt_nominal_orig"] = df["wgt_nominal"].copy()
    df["wgt_nominal"] = np.abs(df["wgt_nominal"])

    print(f"df.dataset.unique(): {df.dataset.unique()}")
    return df

def prepare_dataset(df, ds_dict):
    # Convert dictionary of datasets to a more useful dataframe
    df_info = pd.DataFrame()
    train_samples = []
    for icls, (cls, ds_list) in enumerate(ds_dict.items()):
        for ds in ds_list:
            df_info.loc[ds, "dataset"] = ds
            df_info.loc[ds, "iclass"] = -1
            if cls != "ignore":
                train_samples.append(ds)
                df_info.loc[ds, "class_name"] = cls
                df_info.loc[ds, "iclass"] = icls
    nan_val = -999.0
    df_info = df_info.replace([np.inf, -np.inf], np.nan)
    df_info["iclass"] = df_info["iclass"].fillna(nan_val).astype(int)
    df = df[df.dataset.isin(df_info.dataset.unique())]

    
    # Assign numerical classes to each event
    cls_map = dict(df_info[["dataset", "iclass"]].values)
    print(f"cls_map: {cls_map}")
    cls_name_map = dict(df_info[["dataset", "class_name"]].values)
    df["class"] = df.dataset.map(cls_map)
    df["class_name"] = df.dataset.map(cls_name_map)

    
    # --------------------------------------------------------
    # multiply by dimuon mass resolutions if signal
    # --------------------------------------------------------
    sig_datasets = training_samples["signal"]
    print(f"df.dataset.unique(): {df.dataset.unique()}")
    # if "wgt_flat" not in df.columns:
    #     df['bdt_wgt'] = np.abs(df['wgt_nominal_orig'])
    # else:
    #     df['bdt_wgt'] = np.abs(df['wgt_flat'])
    #     print("taking wgt_flat as bdt wgt!")
    print(f"df['wgt_nominal']: {df['wgt_nominal']}")
    df['bdt_wgt'] = df['wgt_nominal']
    for dataset in sig_datasets:
        dataset_filter = df['dataset']==dataset
        df.loc[dataset_filter,'bdt_wgt'] = df.loc[dataset_filter,'bdt_wgt'] *(1 / df[dataset_filter]['dimuon_ebe_mass_res'])
    # original end -----------------------------------------------

    # -------------------------------------------------
    # normalize sig dataset again to one
    # -------------------------------------------------
    cols = ['dataset', 'bdt_wgt', 'dimuon_ebe_mass_res',]
    mask = df["dataset"].isin(sig_datasets)
    sig_wgt_sum = np.sum(df.loc[mask, "bdt_wgt"])
    print(f'old np.sum(df.loc[mask, "bdt_wgt"]): {sig_wgt_sum}')
    df.loc[mask, "bdt_wgt"] = df.loc[mask, "bdt_wgt"] / sig_wgt_sum

    # print(f"df[cols] after normalization: {df[cols]}")
    print(f'old np.sum(df.loc[mask, "bdt_wgt"]): {sig_wgt_sum}')
    print(f'new np.sum(df.loc[mask, "bdt_wgt"]): {np.sum(df.loc[mask, "bdt_wgt"])}')


    # -------------------------------------------------
    # normalize bkg dataset again to one
    # -------------------------------------------------
    mask = ~df["dataset"].isin(sig_datasets)
    bkg_wgt_sum = np.sum(df.loc[mask, "bdt_wgt"])
    print(f'old np.sum(df.loc[mask, "bdt_wgt"]): {bkg_wgt_sum}')
    df.loc[mask, "bdt_wgt"] = df.loc[mask, "bdt_wgt"] / bkg_wgt_sum

    # print(f"df[cols] after bkg normalization: {df[cols]}")
    print(f'old np.sum(df.loc[mask, "bdt_wgt"]): {bkg_wgt_sum}')
    print(f'new np.sum(df.loc[mask, "bdt_wgt"]): {np.sum(df.loc[mask, "bdt_wgt"])}')

    # -------------------------------------------------
    # increase bdt wgts for bdt to actually learn
    # -------------------------------------------------
    # df['bdt_wgt'] = df['bdt_wgt'] * 10_000
    df['bdt_wgt'] = df['bdt_wgt'] * 100_000 * 100 # NOTE: Why this number???
    # print(f"df[cols] after increase in value: {df[cols]}")
    mask = df["dataset"].isin(sig_datasets)
    # print(f'new signal df.loc[mask, "bdt_wgt"]): {df.loc[mask, "bdt_wgt"]}')
    # print(f'new background (df.loc[mask, "bdt_wgt"]): {df.loc[~mask, "bdt_wgt"]}')
    print(f'new bdt_wgt mean: {np.mean(df["bdt_wgt"])}')
    print(f'new sig bdt_wgt mean: {np.mean(df.loc[mask, "bdt_wgt"])}')
    print(f'new bkg bdt_wgt mean: {np.mean(df.loc[~mask, "bdt_wgt"])}')
    
    #print(df.head)
    # columns_print = ['njets','jj_dPhi','jj_mass_log', 'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta',]
    # columns_print = ['njets','jj_dPhi','jj_mass_log', 'jj_phi', 'jj_pt', 'll_zstar_log', 'mmj1_dEta','jet2_pt']
    # columns2 = ['mmj1_dEta', 'mmj1_dPhi', 'mmj2_dEta', 'mmj2_dPhi', 'mmj_min_dEta', 'mmj_min_dPhi']
    # with open("df.txt", "w") as f:
    #     print(df[columns_print], file=f)
    # with open("df2.txt", "w") as f:
    #     print(df[columns2], file=f)
    # print(df[df['dataset']=="ggh_powheg"].head)
    # print(f"prepare_dataset df: {df["dataset","class"]}")
    return df

def scale_data_withweight(inputs, x_train, x_val, x_eval, df_train, fold_label):
    """
    NOTE: Scaling is not used any more since BDTs don't need it
    """
    # scale data, save the mean and std. This has to be done b4 mixup

    wgt_train = df_train["wgt_nominal_orig"].values
    x_mean = np.average(x_train,axis=0, weights=wgt_train)
    x_std = weighted_std(x_train, wgt_train)

    print(f"x_mean: {x_mean}")
    print(f"x_std: {x_std}")
    
    training_data = (x_train[inputs]-x_mean)/x_std
    validation_data = (x_val[inputs]-x_mean)/x_std
    evaluation_data = (x_eval[inputs]-x_mean)/x_std
    output_path = args["output_path"]
    print(f"scalar output_path: {output_path}/scalers_{name}_{fold_label}")
    print(f"name: {name}")
    save_path = f'{output_path}/bdt_{name}_{year}'
    print(f"scalar save_path: {save_path}/scalers_{name}_{fold_label}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + f"/scalers_{name}_{fold_label}", [x_mean, x_std]) 
    return training_data, validation_data, evaluation_data

def removeStrInColumn(df, str2remove):
    """
    helper function that removes str2remove in df columns if they exist
    """
    new_df = df.rename(
        columns=lambda c: c.replace(str2remove, "") if str2remove in c else c
    )
    return new_df

def getCorrMatrix(df, training_features, save_path=""):
    plt.style.use('default')
    corr_features = training_features + ["dimuon_mass"]
    print(f"getCorrMatrix df.columns: {df.columns}")
    corr_df = df[corr_features]
    corr_df = removeStrInColumn(corr_df, "_nominal")
    corr_matrix = corr_df.corr() 

    if save_path != "": # save as csv and heatmap
        corr_matrix.to_csv(f"{save_path}/correlation_matrix.csv")
        # corr_matrix = corr_matrix.round(2) # round to 2 d.p.
        
        corr = corr_matrix # NOTE: do this instead when loading from csv: corr = corr_matrix.set_index(corr_matrix.columns[0]).astype(float) 
        heatmap = sb.heatmap(corr, fmt=".2f", cmap="coolwarm", annot=True, annot_kws={"fontsize": 12})
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, fontsize = 12)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 45, fontsize = 12)
        
        plt.title("BDT input features + dimuon mass correlation matrix")
        # plt.tight_layout()
        fig = heatmap.get_figure()
        fig.set_size_inches(16,12)
        fig.savefig(f"{save_path}/correlation_matrix.pdf", bbox_inches='tight', pad_inches=0)

    plt.style.use(hep.style.CMS)
    # raise ValueError
    return corr_matrix

def load_best_params_for_fold(meta_path, fold_idx):
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    key = f"hyperparameter_search_fold{fold_idx}"
    if key not in metadata:
        raise KeyError(f"{key} not found in {meta_path}")

    best_params = metadata[key].get("best_params", None)
    if best_params is None:
        raise ValueError(f"best_params missing for {key} in {meta_path}")

    return best_params

def classifier_train(df, args, training_samples, training_features, random_seed_val: int, save_path:str, do_hyperparam_search=False, n_trials=25):
    print(f"random_seed_val: {random_seed_val}")

    nfolds = 4
    print(f"Training features: {training_features}")
    year = args['year']
    name = args['name']
    print(f"year: {year}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save training features as json for readability
    with open(f'{save_path}/training_features.json', 'w') as file:
        json.dump(training_features, file, indent=4)
    # get the overal correlation matrix
    corr_matrix = getCorrMatrix(df, training_features, save_path=save_path)

    # ------------------- NEW: initialize ONE metadata dict -------------------
    meta_path = os.path.join(save_path, "training_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:    
        metadata = {
            "pipeline": {
                "preprocessor_version": "v1.1",
                "git_commit": get_git_commit(),
                "git_state": get_git_state(save_path),
            },        
            "run": {
                "timestamp_utc": _utc_now_iso(),
                "args": dict(args),
                "random_seed_val": int(random_seed_val),
                "nfolds": int(nfolds),
                "training_features": list(training_features),
                "training_samples": training_samples,
                "df_global": {
                    "n_rows": int(len(df)),
                    "columns": list(df.columns),
                    "datasets": sorted([str(x) for x in df["dataset"].unique()]) if "dataset" in df.columns else [],
                    "classes": sorted([int(x) for x in df["class"].unique()]) if "class" in df.columns else [],
                },
            },
        }
        for i in range(nfolds):
            metadata[f"folds_{i}"] = {}
        for i in range(nfolds):
            metadata[f"hyperparameter_search_fold{i}"] = {}
        _write_metadata_json(meta_path, metadata)
    # ------------------------------------------------------------------------
    
    for i in range(nfolds):
        if args['year']=='':
            label = f"allyears_{args['label']}_{i}"
        else:
            label = f"{args['year']}_{args['label']}{i}"
        
        train_folds = [(i+f)%nfolds for f in [0,1]]
        val_folds = [(i+f)%nfolds for f in [2]]
        eval_folds = [(i+f)%nfolds for f in [3]]

        print(f"Train classifier #{i+1} out of {nfolds}")
        print(f"Training folds: {train_folds}")
        print(f"Validation folds: {val_folds}")
        print(f"Evaluation folds: {eval_folds}")
        print(f"Samples used: {df.dataset.unique()}")
        
        train_filter = df.event.mod(nfolds).isin(train_folds)
        val_filter = df.event.mod(nfolds).isin(val_folds)
        eval_filter = df.event.mod(nfolds).isin(eval_folds)
        
        df_train = df[train_filter]
        df_val = df[val_filter]
        df_eval = df[eval_filter]

        
        x_train = df_train[training_features]
        y_train = df_train['class']
        x_val = df_val[training_features]
        x_eval = df_eval[training_features]
        y_val = df_val['class']
        y_eval = df_eval['class']

        # print(f"y_train: {y_train}")
        # print(f"y_val: {y_val}")
        # print(f"y_eval: {y_eval}")
        # print(f"df_train: {df_train.head()}")
        
        # original start -------------------------------------------------------
        classes = {
            0 : 'background',
            1 : 'signal',
        }
        print(f"classes: {classes}")
        
        for icls, cls in classes.items():
            print(f"icls: {icls}")
            train_evts = len(y_train[y_train==icls])
            df_train.loc[y_train==icls,'cls_avg_wgt'] = df_train.loc[y_train==icls,'wgt_nominal'].values.mean()
            df_val.loc[y_val==icls,'cls_avg_wgt'] = df_val.loc[y_val==icls,'wgt_nominal'].values.mean()
            df_eval.loc[y_eval==icls,'cls_avg_wgt'] = df_eval.loc[y_eval==icls,'wgt_nominal'].values.mean()
            print(f"{train_evts} training events in class {cls}")
        # original end -------------------------------------------------------

            
        xp_train = x_train[training_features].values
        xp_val = x_val[training_features].values
        xp_eval = x_eval[training_features].values
        y_train = y_train.values
        y_val = y_val.values
        y_eval = y_eval.values

        print(f"xp_train.shape: {xp_train.shape}")
        print(f"xp_val.shape: {xp_val.shape}")
        print(f"xp_eval.shape: {xp_eval.shape}")

        # NOTE: bdt_wgt = (1/ebe)*(Class weight imbalance weight)*(Mass decorelation weight)
        w_train = df_train['bdt_wgt'].values

        weight_nom_train = df_train['wgt_nominal_orig'].values
        weight_nom_val = df_val['wgt_nominal_orig'].values
        weight_nom_eval = df_eval['wgt_nominal_orig'].values
        
        np.random.seed(random_seed_val)
        
        shuf_ind_tr = np.arange(len(xp_train))
        np.random.shuffle(shuf_ind_tr)
        shuf_ind_val = np.arange(len(xp_val))
        np.random.shuffle(shuf_ind_val)
        shuf_ind_eval = np.arange(len(xp_eval))
        np.random.shuffle(shuf_ind_eval)
        xp_train = xp_train[shuf_ind_tr]
        xp_val = xp_val[shuf_ind_val]
        y_train = y_train[shuf_ind_tr]
        y_val = y_val[shuf_ind_val]

        
        xp_eval = xp_eval[shuf_ind_eval]
        y_eval = y_eval[shuf_ind_eval]

        weight_nom_train = weight_nom_train[shuf_ind_tr]
        weight_nom_val = weight_nom_val[shuf_ind_val]
        weight_nom_eval = weight_nom_eval[shuf_ind_eval]
        #print(np.isnan(xp_train).any())
        #print(np.isnan(y_train).any())
        #print(np.isinf(xp_train).any())
        #print(np.isinf(y_train).any())
        #print(np.isfinite(x_train).all())
        #print(np.isfinite(y_train).all())
        
        w_train = w_train[shuf_ind_tr]

        verbosity=2
        device = get_xgb_device()
        print(f"\n\n====> device: {device}")
        # -----------------------------------------
        # Do hyperparameter tuning if asked
        # instead of normal fitting
        # -----------------------------------------
        if do_hyperparam_search:
            import optuna

            from modules.hyperparamOptim import objective
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed_val))
            study.optimize(lambda trial: objective(trial, xp_train, xp_val, y_train, y_val, w_train, weight_nom_val, random_seed=random_seed_val), n_trials=n_trials)
            print(f"Fold {i} Best AUC: {study.best_value}")
            print(f"Fold {i} Best params: {study.best_params}")

            metadata[f"hyperparameter_search_fold{i}"] = {
                "timestamp": _utc_now_iso(),

                "sampler": "TPESampler",
                "direction": "maximize",

                "n_trials": len(study.trials),

                "best_value": float(study.best_value),
                "best_params": study.best_params,

                "best_trial_number": study.best_trial.number,

                "trials": [
                    {
                        "trial_number": t.number,
                        "value": float(t.value) if t.value is not None else None,
                        "params": t.params,
                        "state": str(t.state)
                    }
                    for t in study.trials
                ]
            }
            _write_metadata_json(meta_path, metadata)            
            continue

        #--------------------------------------------------   
        # BDT hyparameter setup
        #--------------------------------------------------           
        # AN-19-124 p 45: "a correction factor is introduced to ensure that the same amount of background events are expected when either negative weighted events are discarded or they are considered with a positive weight"
        tuned_params = load_best_params_for_fold(meta_path, i)
        tuned_params = dict(tuned_params)
        tuned_params.update({
            "tree_method" : 'hist',
            "device": device,
            "eval_metric": ["logloss", "error", "auc"],
            "n_jobs" : 30,
            "early_stopping_rounds" : 15,
            "verbosity" : verbosity,
            "random_state" : random_seed_val,
        })
        model = XGBClassifier(**tuned_params)
        
        print(model)
        print(f"negative w_train: {w_train[w_train <0]}")

        eval_set = [(xp_train, y_train), (xp_val, y_val)] # Last used
        print(f"has_bad_values(w_train): {has_bad_values(w_train)}")
        print(f"has_bad_values(xp_train): {has_bad_values(xp_train)}")
        print(f"has_bad_values(y_train): {has_bad_values(y_train)}")
        print(f"has_bad_values(xp_val): {has_bad_values(xp_val)}")
        print(f"has_bad_values(y_val): {has_bad_values(y_val)}")
        print(f"y_train unqiue: {np.unique(y_train)}")
        print(f"y_val unqiue: {np.unique(y_val)}")


        # ------------------- NEW: per-fold prefit metadata -------------------
        fold_meta = {
            "label": label,
            "fold_index": int(i),
            "timestamp_utc_prefit": _utc_now_iso(),
            "splits": {"train_folds": train_folds, "val_folds": val_folds, "eval_folds": eval_folds},
            "x_shapes": {"train": list(xp_train.shape), "val": list(xp_val.shape), "eval": list(xp_eval.shape)},
            "y_unique": {
                "train": [int(x) for x in np.unique(y_train)],
                "val":   [int(x) for x in np.unique(y_val)],
                "eval":  [int(x) for x in np.unique(y_eval)],
            },
            "composition": {
                "train": _df_class_breakdown(df_train),
                "val":   _df_class_breakdown(df_val),
                "eval":  _df_class_breakdown(df_eval),
            },
            "weights": {
                "w_train": _array_stats(w_train, "w_train"),
                "weight_nom_train": _array_stats(weight_nom_train, "weight_nom_train"),
                "weight_nom_val":   _array_stats(weight_nom_val, "weight_nom_val"),
                "weight_nom_eval":  _array_stats(weight_nom_eval, "weight_nom_eval"),
            },
            "tuned_params": tuned_params,
            "fit_call": {
                "sample_weight": True,
                "eval_set": ["train", "val"],
                "verbose": False,
                # set to True if you use it
                "sample_weight_eval_set": False,
                # "sample_weight_eval_set_names": ["w_train", "weight_nom_val"],
                "sample_weight_eval_set_names": [],
            },
        }
        # --------------------------------------------------------------------

        # -----------------------------------------
        # Do normal BDT fitting 
        # -----------------------------------------
        # fit options: fit(X, y, *, 
        # sample_weight=None, base_margin=None, 
        # eval_set=None, verbose=True, 
        # xgb_model=None, 
        # sample_weight_eval_set=None, base_margin_eval_set=None, 
        # feature_weights=None)
        model.fit(xp_train, y_train, sample_weight = w_train, eval_set=eval_set, verbose=False)
        # model.fit(xp_train, y_train, sample_weight = w_train, eval_set=eval_set, verbose=False, sample_weight_eval_set=[w_train, weight_nom_val])
        
        y_pred = model.predict_proba(xp_val)[:, 1].ravel()
        y_pred_train = model.predict_proba(xp_train)[:, 1].ravel()
        y_eval_pred = model.predict_proba(xp_eval)[:, 1].ravel()        
        print("y_pred_______________________________________________________________")
        print("y_pred_______________________________________________________________")
        print("y_pred_______________________________________________________________")
        # print(f"y_pred: {y_pred}")
        print("y_pred_______________________________________________________________")
        print("y_pred_______________________________________________________________")
        print("y_pred_______________________________________________________________")
        # print(f"y_val: {y_val}")
        # original start ------------------------------------------------------------------------------


        # output shape dist start --------------------------------------------------------------------------
        fig, ax_main = plt.subplots()
        # Define custom bins from 0 to 1
        binning = np.linspace(0, 1, 31)  # 30 bins between 0 and 1

        # get the distributions
        is_bkg_train = y_train.ravel() == 0
        y_pred_train_bkg = y_pred_train[is_bkg_train]
        weight_nom_train_bkg = weight_nom_train[is_bkg_train]
        hist_train_bkg, _ = generate_normalized_histogram(y_pred_train_bkg, weight_nom_train_bkg, binning)

        is_sig_train = y_train.ravel() == 1
        y_pred_train_sig = y_pred_train[is_sig_train]
        weight_nom_train_sig = weight_nom_train[is_sig_train]
        hist_train_sig, _ = generate_normalized_histogram(y_pred_train_sig, weight_nom_train_sig, binning)
        
        is_bkg_val = y_val.ravel() == 0
        y_pred_val_bkg = y_pred[is_bkg_val]
        weight_nom_val_bkg = weight_nom_val[is_bkg_val]
        hist_val_bkg, _ = generate_normalized_histogram(y_pred_val_bkg, weight_nom_val_bkg, binning)

        is_sig_val = y_val.ravel() == 1
        y_pred_val_sig = y_pred[is_sig_val]
        weight_nom_val_sig = weight_nom_val[is_sig_val]
        hist_val_sig, _ = generate_normalized_histogram(y_pred_val_sig, weight_nom_val_sig, binning)

        is_bkg_eval = y_eval.ravel() == 0
        y_pred_eval_bkg = y_eval_pred[is_bkg_eval]
        weight_nom_eval_bkg = weight_nom_eval[is_bkg_eval]
        hist_eval_bkg, _ = generate_normalized_histogram(y_pred_eval_bkg, weight_nom_eval_bkg, binning)

        is_sig_eval = y_eval.ravel() == 1
        y_pred_eval_sig = y_eval_pred[is_sig_eval]
        weight_nom_eval_sig = weight_nom_eval[is_sig_eval]
        hist_eval_sig, _ = generate_normalized_histogram(y_pred_eval_sig, weight_nom_eval_sig, binning)
        
        
        hep.histplot(
            hist_train_bkg, 
            bins=binning, 
            stack=False, 
            histtype='step', 
            # color='blue', 
            label='Background train', 
            ax=ax_main,
        )
        hep.histplot(
            hist_val_bkg, 
            bins=binning, 
            stack=False, 
            histtype='step', 
            # color='green', 
            label='Background Validation', 
            ax=ax_main,
        )
        hep.histplot(
            hist_eval_bkg, 
            bins=binning, 
            stack=False, 
            histtype='step', 
            # color='blue', 
            label='Background Eval', 
            ax=ax_main,
        )
        hep.histplot(
            hist_train_sig, 
            bins=binning, 
            stack=False, 
            histtype='step', 
            # color='red', 
            label='Signal train', 
            ax=ax_main,
        )
        hep.histplot(
            hist_val_sig, 
            bins=binning, 
            stack=False, 
            histtype='step', 
            # color='blue', 
            label='Signal Validation', 
            ax=ax_main,
        )
        hep.histplot(
            hist_eval_sig, 
            bins=binning, 
            stack=False, 
            histtype='step', 
            # color='blue', 
            label='Signal Eval', 
            ax=ax_main,
        )
        
        
        # Add labels, title, and legend
        ax_main.set_xlabel('BDT Score')
        ax_main.set_ylabel('A.U.')
        ax_main.legend()
        ax_main.set_title('BDT Score distribution')
        
        fig.savefig(f"{save_path}/BDT_Score_{label}.png")
        ax_main.clear()
        plt.cla()
        plt.clf()

        # output shape dist end --------------------------------------------------------------------------

        
        # -------------------------------------------
        # GoF test
        # -------------------------------------------
        gof_save_path = save_path
        # print(f"weight_nom_val_sig: {weight_nom_val_sig}")
        # print(f"weight_nom_train_sig: {weight_nom_train_sig}")
        print(f"weight_nom_val_sig: {type(weight_nom_val_sig)}")
        print(f"weight_nom_train_sig: {type(weight_nom_train_sig)}")
        print(f"weight_nom_val_sig: {len(weight_nom_val_sig)}")
        print(f"weight_nom_train_sig: {len(weight_nom_train_sig)}")
        
        # we compare validation distribution with evaluation distribution to see if there's any over-training
        getGOF_KS_bdt(hist_eval_sig, hist_val_sig, weight_nom_val_sig, binning, gof_save_path, label)

        # -------------------------------------------
        # Log scale ROC curve
        # -------------------------------------------
        roc_data_dict = {
            "y_train": y_train.ravel(),
            "y_pred_train": y_pred_train,
            "weight_nom_train": weight_nom_train,
        
            "y_val": y_val.ravel(),
            "y_pred": y_pred,
            "weight_nom_val": weight_nom_val,
        
            "y_eval": y_eval.ravel(),
            "y_eval_pred": y_eval_pred,
            "weight_nom_eval": weight_nom_eval,
        }

        auc_df_NoClsBalance = fullROC_operations(fig, roc_data_dict, save_path, year, label, doClassBalance=False)
        auc_df_ClsBalance = fullROC_operations(fig, roc_data_dict, save_path, year, label, doClassBalance=True)        
        
        # do fig 6.5 start --------------------------------------------------------------
        get6_5(y_eval.ravel(), y_eval_pred, weight_nom_eval, save_path, f"eval_{label}")
        get6_5(y_val.ravel(), y_pred, weight_nom_val, save_path, f"val_{label}")
        # do fig 6.5 end --------------------------------------------------------------



        # -----------------------------
        # Retrieve evaluation results
        # -----------------------------
        results = model.evals_result()
        epochs = len(results["validation_0"]["logloss"])
        plot_x_axis = range(0, epochs)
        
        # ------------------- NEW: per-fold postfit metadata -------------------
        # Make evals_result JSONable
        clean_results = {}
        for k, v in results.items():
            clean_results[str(k)] = {}
            for mk, mv in v.items():
                clean_results[str(k)][str(mk)] = [float(x) for x in mv]

        fold_meta.update({
            "timestamp_utc_postfit": _utc_now_iso(),
            "best_iteration": int(model.best_iteration) if model.best_iteration is not None else None,
            "best_score": float(model.best_score) if model.best_score is not None else None,
            "evals_result": clean_results,
            "auc_df_NoClsBalance":  _df_to_dict_safe(auc_df_NoClsBalance),
            "auc_df_ClsBalance":  _df_to_dict_safe(auc_df_ClsBalance),
        })

        metadata[f"folds_{i}"][label] = fold_meta
        _write_metadata_json(meta_path, metadata)
        # --------------------------------------------------------------------

        # -----------------------------
        # Plot training vs. validation loss
        # -----------------------------
        if "logloss" in results["validation_0"] and "logloss" in results["validation_1"]:
            plt.clf()
            train_loss = results["validation_0"]["logloss"]
            val_loss = results["validation_1"]["logloss"]
            plt.plot(plot_x_axis, train_loss, label="Train ")
            plt.plot(plot_x_axis, val_loss, label="Validation")

            plt.xlabel("Boosting Round")
            plt.ylabel("Log Loss")
            plt.title("XGBoost Training vs. Validation Loss")
            perf_text = plt.Line2D([], [], color='none', label=f'Best iteration: {model.best_iteration} \n Best score: {model.best_score:.5f}')
            plt.legend(handles=[perf_text])
            plt.savefig(f"{save_path}/loss_{label}.png")
            
        # -----------------------------
        # Plot training vs. AUC
        # -----------------------------        
        if "auc" in results["validation_0"] and "auc" in results["validation_1"]:
            plt.clf()
            plt.plot(plot_x_axis, results['validation_0']['auc'], label='Train')
            plt.plot(plot_x_axis, results['validation_1']['auc'], label='Validation')
            plt.xlabel("Boosting Round")
            plt.ylabel('AUC')
            plt.title('XGBoost AUC')
            plt.legend()
            plt.savefig(f"{save_path}/auc_{label}.png")

        # -----------------------------
        # Plot training vs. classification error
        # -----------------------------      
        if "error" in results["validation_0"] and "error" in results["validation_1"]:
            plt.clf()
            plt.plot(plot_x_axis, results['validation_0']['error'], label='Train')
            plt.plot(plot_x_axis, results['validation_1']['error'], label='Validation')
            plt.xlabel("Boosting Round")
            plt.ylabel('Classification Error')
            plt.title('XGBoost Classification Error')
            plt.legend()
            plt.savefig(f"{save_path}/classificationError_{label}.png")
        # -----------------------------
        # 6. Check the best iteration
        # -----------------------------
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best validation loss: {model.best_score:.5f}")
        
        csv_savepath = f"{save_path}/loss_{label}.csv"
        loss_df = pd.DataFrame({
            "epoch" : plot_x_axis,
            "train_loss" : train_loss,
            "val_loss" : val_loss,
        })
        loss_df.to_csv(csv_savepath)
        

        labels = [feat.replace("_nominal","") for feat in training_features]
        model.get_booster().feature_names = labels # set my training features as feature names
        
        # plot trees
        plot_tree(model)
        plt.savefig(f"{save_path}/TreePlot_{i}.png",dpi=400)


        feature_important = model.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        score_name = "Weight Score"
        data = pd.DataFrame(data=values, index=keys, columns=[score_name]).sort_values(by = score_name, ascending=True)
        data["Normalized Score"] = data[score_name] / data[score_name].sum()
        data.to_csv(f"{save_path}/BDT_FeatureImportance_{label}_byWeight.csv")
        data.nlargest(50, columns=score_name).plot(kind='barh', figsize = (20,10))
        data.plot(kind='barh', figsize = (20,10))
        plt.savefig(f"{save_path}/BDT_FeatureImportance_{label}_byWeight.png")

        feature_important = model.get_booster().get_score(importance_type='gain')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        score_name = "Gain Score"
        data = pd.DataFrame(data=values, index=keys, columns=[score_name]).sort_values(by = score_name, ascending=True)
        data["Normalized Score"] = data[score_name] / data[score_name].sum()
        data.to_csv(f"{save_path}/BDT_FeatureImportance_{label}_byGain.csv")
        data.nlargest(50, columns=score_name).plot(kind='barh', figsize = (20,10))
        data.plot(kind='barh', figsize = (20,10))
        plt.savefig(f"{save_path}/BDT_FeatureImportance_{label}_byGain.png")

        
        output_path = args["output_path"]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_fname = (f"{save_path}/{name}_{label}.pkl")
        pickle.dump(model, open(model_fname, "wb"))
        print ("wrote model to",model_fname)
            

def evaluation(df, args):
    if df.shape[0]==0: return df
    if args['bdt']:
        if args['do_massscan']:
            mass_shift = args['mass']-125.0
        add_year = args['evaluate_allyears_dnn']
        #df = prepare_features(df, args, add_year)
        df['bdt_score'] = 0
        nfolds = 4
        for i in range(nfolds):    
            if args['evaluate_allyears_dnn']:
                label = f"allyears_{args['label']}_{i}"
            else:
                label = f"{args['year']}_{args['label']}{i}"
            
            
            train_folds = [(i+f)%nfolds for f in [0,1]]
            val_folds = [(i+f)%nfolds for f in [2]]
            eval_folds = [(i+f)%nfolds for f in [3]]
            
            eval_filter = df.event.mod(nfolds).isin(eval_folds)

            # print(f"train_folds: {train_folds}")
            # print(f"val_folds: {val_folds}")
            # print(f"eval_folds: {eval_folds}")

            # eval_label = f"{args['year']}_{args['label']}{eval_folds[0]}"
            eval_label = f"{args['year']}_{args['label']}{i}"
            print(f"eval_label: {eval_label}")
            
            # scalers_path = f"{output_path}/{name}_{year}/scalers_{name}_{eval_label}.npy"
            # start_path = "/depot/cms/hmm/copperhead/trained_models/"
            output_path = args["output_path"]
            print(f"output_path: {output_path}")
            # scalers_path = f"{output_path}/bdt_{name}_{year}/scalers_{name}_{eval_label}.npy"

            # print(f"scalers_path: {scalers_path}")
            #scalers_path = f'output/trained_models_nest10000/scalers_{label}.npy'
            # scalers = np.load(scalers_path)

            model_path = f"{output_path}/bdt_{name}_{year}/{name}_{label}.pkl"
            #model_path = f'output/trained_models_nest10000/BDT_model_earlystop50_{label}.pkl'
            bdt_model = pickle.load(open(model_path, "rb"))
            print(f"bdt_model.classes_: {bdt_model.classes_}")
            df_i = df[eval_filter]
            #if args['r']!='h-peak':
            #    df_i['dimuon_mass'] = 125.
            #if args['do_massscan']:
            #    df_i['dimuon_mass'] = df_i['dimuon_mass']+mass_shift
            #print("Scaler 0 ",scalers[0])
            #print("Scaler 1 ",scalers[1])
            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",df_i.head())
            # df_i = (df_i[training_features]-scalers[0])/scalers[1]
            #print("************************************************************",df_i.head())
            prediction_sig = np.array(bdt_model.predict_proba(df_i.values)[:, 1]).ravel()
            prediction_bkg = np.array(bdt_model.predict_proba(df_i.values)[:, 0]).ravel()
            fig1, ax1 = plt.subplots(1,1)
            plt.hist(prediction_sig, bins=50, alpha=0.5, color='blue', label='Validation Sig')
            plt.hist(prediction_bkg, bins=50, alpha=0.5, color='red', label='Validation BKG')


            ax1.legend(loc="upper right")
            save_path = f"output/bdt_{name}_{year}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig1.savefig(f"{save_path}/Validation_{label}.png")
            

            # df.loc[eval_filter,'bdt_score'] = prediction
    return df