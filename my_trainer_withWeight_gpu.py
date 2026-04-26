import argparse
import glob
import os
import time

import awkward as ak
import dask_awkward as dak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from distributed import Client
import json
from pathlib import Path

from rich import print

from modules.variables import training_features, training_samples

from modules.workflow import (
    classifier_train,
    convert2df,
    prepare_dataset,
    prepare_features,
)

from modules.utils import (
    apply_gghChannelSelection,
    reweightMassToFlat,
    reweightMassToTargetDist_workflow,
    split_into_n_parts,
    PairNAnnhilateNegWgt_inChunks,
)
from modules.utils_logger import logger

plt.style.use(hep.style.CMS)


def resolve_stage1_base(load_root, year):
    load_root = load_root.rstrip("/")
    if year == "2016":
        stage1_dir = "compacted" if glob.glob(f"{load_root}/{year}*/compacted") else "f1_0"
        return f"{load_root}/{year}*/{stage1_dir}"
    if year == "all":
        stage1_dir = "compacted" if glob.glob(f"{load_root}/*/compacted") else "f1_0"
        return f"{load_root}/*/{stage1_dir}"

    compacted_dir = Path(load_root) / year / "compacted"
    stage1_dir = "compacted" if compacted_dir.exists() else "f1_0"
    return f"{load_root}/{year}/{stage1_dir}"

def get_cache_paths(save_path, year, negWgtHandling, mass_decorrelation_strat):
    cache_dir = Path(save_path) / "cached_training_df"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{year}__negWgt-{negWgtHandling}__massDecorr-{mass_decorrelation_strat}"
    parquet_path = cache_dir / f"training_df_{tag}.parquet"
    meta_path = cache_dir / f"training_df_{tag}.json"
    return parquet_path, meta_path


def save_cached_training_df(df, parquet_path, meta_path, extra_meta=None):
    df.to_parquet(parquet_path, index=False)

    meta = {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "datasets": sorted(df["dataset"].unique().tolist()) if "dataset" in df.columns else [],
    }
    if extra_meta is not None:
        meta.update(extra_meta)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved cached dataframe to: {parquet_path}")
    print(f"[INFO] Saved cache metadata to: {meta_path}")


def load_cached_training_df(parquet_path, meta_path=None):
    print(f"[INFO] Loading cached dataframe from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if meta_path is not None and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"[INFO] Cache metadata: n_rows={meta.get('n_rows')}, datasets={meta.get('datasets')}")

    return df


def prepare_features_from_df(df, features, variation="nominal"):
    features_var = []
    for trf in features:
        if "soft" in trf:
            variation_current = "nominal"
        else:
            variation_current = variation

        if f"{trf}_{variation_current}" in df.columns:
            features_var.append(f"{trf}_{variation_current}")
        elif trf in df.columns:
            features_var.append(trf)
        else:
            print(f"[WARN] Variable {trf} not found in cached dataframe columns!")
    return features_var


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "-y",
    "--year",
    dest="year",
    default="2018",
    action="store",
    help="Year to process (2016preVFP, 2016postVFP, 2017 or 2018)",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        default="test",
        action="store",
        help="Name of classifier",
    )
    parser.add_argument(
        "--vbf",
        dest="is_vbf",
        default=False, 
        action=argparse.BooleanOptionalAction,
        help="If true we filter out for VBF production mode category, if not, cut for ggH category",
    )
    parser.add_argument(
    "-load",
    "--load_path",
    dest="load_path",
    default="/depot/cms/users/yun79/results/stage1/DNN_test2/",
    action="store",
    help="Year to process (2016preVFP, 2016postVFP, 2017 or 2018)",
    )
    parser.add_argument(
    "-param_search",
    "--do_hyperparam_search",
    dest="do_hyperparam_search",
    type=int,
    default=0,
    # action="store",
    help="integer flag to do hyperparam search. If zero, then do not do it",
    )
    parser.add_argument(
        "--n_trials",
        "--n_trials",
        type=int,
        default=21,
        help="Number of trials for the Bayseian optimization",
    )
    parser.add_argument(
    "--massDeCorrStrat",
    dest="mass_decorrelation_strat",
    default="default",
    action="store",
    help="Dimuon mass decorrelation method for training. Available options are: default (do nothing), peking, targetZpeakMass, flatDist",
    )
    parser.add_argument(
    "--negWgtHandling",
    dest="negWgtHandling",
    default="pairAndAnnhilate",
    action="store",
    help="algorithm for handling the negative weight during training. The options are: pairAndAnnhilate, takeAbsWgts, removeNegWgts",
    )
    parser.add_argument(
        "--overwrite_cached_df",
        dest="overwrite_cached_df",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, rebuild and overwrite the cached training dataframe.",
    )
    parser.add_argument(
        "--target_dist_path",
        dest="target_dist_path",
        default=None,
        action="store",
        help="Reference parquet used for target-shape mass decorrelation modes.",
    )
    sysargs = parser.parse_args()
    year = sysargs.year
    name = sysargs.name
    negWgtHandling = sysargs.negWgtHandling
    # mass_decorrelation_strat = sysargs.mass_decorrelation_strat
    args = {
        "year": year,
        "name": name,
        # "mass_decorrelation_strat": mass_decorrelation_strat,
        "do_massscan": False,
        "evaluate_allyears_dnn": False,
        "output_path": "output/trained_MVAs",
        "label": "",
        # "do_hyperparam_search": !(sysargs.do_hyperparam_search==0), # if zero, then do not do hyperparam search
    }
    do_hyperparam_search = (sysargs.do_hyperparam_search!=0) # if zero, then do not do hyperparam search
    print(f"do_hyperparam_search: {do_hyperparam_search}")
    start_time = time.time()

    save_path = f"output/bdt_{name}_{year}"
    os.makedirs(save_path, exist_ok=True)

    cache_parquet_path, cache_meta_path = get_cache_paths(
        save_path=save_path,
        year=year,
        negWgtHandling=negWgtHandling,
        mass_decorrelation_strat=sysargs.mass_decorrelation_strat,
    )

    # client =  Client(n_workers=31,  threads_per_worker=1, processes=True, memory_limit='12 GiB') 
    client =  Client(n_workers=10,  threads_per_worker=1, processes=True, memory_limit='25 GiB') 

    
    load_path = resolve_stage1_base(sysargs.load_path, year)
    print(f"load_path: {load_path}")
    sample_l = training_samples["background"] + training_samples["signal"]
    is_UL = True
    

    fields2load = [ 
            "dimuon_mass",
            "jj_mass_nominal",
            "jj_dEta_nominal",
            "jet1_pt_nominal",
            "nBtagLoose_nominal",
            "nBtagMedium_nominal",
            # "mmj1_dEta_nominal",
            # "mmj2_dEta_nominal",
            # "mmj1_dPhi_nominal",
            # "mmj2_dPhi_nominal",
            "wgt_nominal",
            "dimuon_ebe_mass_res",
            "event",
            "mu1_pt",
            "mu2_pt",
            "year",
            "njets_nominal",
        ] 

    
    fields2load = list(set(fields2load + training_features)) # remove redundancies
    # load data to memory using compute()
    
    
    training_features_prepared = training_features[:]  # default fallback

    if cache_parquet_path.exists() and (not sysargs.overwrite_cached_df):
        df_total = load_cached_training_df(cache_parquet_path, cache_meta_path)
        training_features_prepared = prepare_features_from_df(df_total, training_features)
    else:
        df_l = []
        print(f"sample_l: {sample_l}")
        print(f"training_features: {training_features}")
        print(f"load_path: {load_path}")

        for sample in sample_l:
            print(f"running sample {sample}")
            parquet_path = load_path + f"/{sample}/*/*.parquet"
            print(f"parquet_path: {parquet_path}")
            filelist_big = glob.glob(parquet_path)

            # FIXME --------------------
            # print(f"===> year: {year}")
            # print(f"===> sysargs.load_path: {sysargs.load_path}")
            # if (year == "all") or (year == "2024"):
            #     if sample=="dyTo2L_M-50_incl": 
            #         load_path_exception = f"{sysargs.load_path}/2024/f1_0" 
            #         parquet_path = load_path_exception+f"/dyTo2Mu_M-50_aMCatNLO/*/*.parquet"
            #         filelist_big += glob.glob(parquet_path)

            if "dy" in sample:
                n_parts = 4
            else:
                n_parts = 1
            filelist_l = split_into_n_parts(filelist_big, n_parts)

            for filelist in filelist_l:
                try:
                    if len(filelist) == 0:
                        print(f"[WARN] No parquet files found for {sample} at: {parquet_path}. Skipping.")
                        continue  # skip this sample                

                    zip_sample = dak.from_parquet(filelist)

                    fields2load_prepared = prepare_features(zip_sample, fields2load) # add variation to the name
                    training_features_prepared = prepare_features(zip_sample, training_features) # do the same thing to training features

                    # print(f"fields2load after: {fields2load_prepared}")
                    zip_sample = ak.zip({
                        field: zip_sample[field] for field in fields2load_prepared
                    })
                    zip_sample = apply_gghChannelSelection(zip_sample)

                except Exception as error:
                    raise FileNotFoundError(f"Parquet for {sample} not found with error {error}. skipping!")

                df_sample = convert2df(zip_sample, sample)

                if negWgtHandling == "pairAndAnnhilate":
                    print("Applying pairAndAnnhilate!")
                    max_num_rows = 80_000
                    wgt_col = "wgt_nominal_orig"
                    df_sample = PairNAnnhilateNegWgt_inChunks(df_sample, max_num_rows=max_num_rows)
                    print(f"any neg wgts: {np.any(df_sample[wgt_col] < 0)}")

                elif negWgtHandling == "takeAbsWgts":
                    print("Applying takeAbsWgts!")
                    # do nothing convert2df already takes the absolute value of wgt_notminal_orig
                    pass

                elif negWgtHandling == "removeNegWgts":
                    print("Applying removeNegWgts!")
                    wgt_col = "wgt_nominal_orig"
                    wgt_filter = df_sample[wgt_col] >= 0
                    df_sample = df_sample[wgt_filter]
                    print(f"any neg wgts: {np.any(df_sample[wgt_col] < 0)}")

                else:
                    raise ValueError("Error: Unsupported negative weight handling method!")

                df_l.append(df_sample)

        if not df_l:
            raise FileNotFoundError(
                f"No parquet files were loaded for training from {load_path}. "
                "Check --load_path, year, and whether the stage1 output exists."
            )

        df_total = pd.concat(df_l, ignore_index=True)
        del df_l

        print(f"df_total.dataset.unique(): {df_total.dataset.unique()}")

        sig_datasets = training_samples["signal"]

        if sysargs.mass_decorrelation_strat == "peking":
            print("Peking decorrlation method!")
            df_total = reweightMassToFlat(df_total, sig_datasets, save_path)

        elif sysargs.mass_decorrelation_strat == "targetZpeakMass":
            print("targetZpeakMass decorrlation method!")
            df_total = reweightMassToTargetDist_workflow(
                df_total, sig_datasets, save_path, target_dist_load_path=sysargs.target_dist_path
            )

        elif sysargs.mass_decorrelation_strat == "targetHpeakMass":
            print("targetHpeakMass decorrlation method!")
            dy_target_mass_centre = 125
            nbins = 20
            df_total = reweightMassToTargetDist_workflow(
                df_total, sig_datasets, save_path,
                nbins=nbins, target_mass_centre=dy_target_mass_centre,
                target_dist_load_path=sysargs.target_dist_path,
            )

        elif sysargs.mass_decorrelation_strat == "targetHsidebandMass":
            print("targetHpeakMass decorrlation method!")
            dy_target_mass_centre = 105
            nbins = 40
            df_total = reweightMassToTargetDist_workflow(
                df_total, sig_datasets, save_path,
                nbins=nbins, target_mass_centre=dy_target_mass_centre,
                target_dist_load_path=sysargs.target_dist_path,
            )

        elif sysargs.mass_decorrelation_strat == "flatDist":
            print("flatDist decorrlation method!")
            dy_target_mass_centre = "flat"
            df_total = reweightMassToTargetDist_workflow(
                df_total, sig_datasets, save_path,
                target_mass_centre=dy_target_mass_centre,
                target_dist_load_path=sysargs.target_dist_path,
            )

        elif sysargs.mass_decorrelation_strat == "sinusoidalDist":
            print("sinusoidal decorrlation method!")
            dy_target_mass_centre = "sinusoidal"
            df_total = reweightMassToTargetDist_workflow(
                df_total, sig_datasets, save_path,
                target_mass_centre=dy_target_mass_centre,
                target_dist_load_path=sysargs.target_dist_path,
            )

        training_features_prepared = prepare_features_from_df(df_total, training_features)
        save_cached_training_df(
            df_total,
            cache_parquet_path,
            cache_meta_path,
            extra_meta={
                "year": year,
                "negWgtHandling": negWgtHandling,
                "mass_decorrelation_strat": sysargs.mass_decorrelation_strat,
                "training_features": training_features_prepared,
            },
        )

    # one hot-encode start ----------------------------------------------------
    do_one_hot_encode = False
    if do_one_hot_encode:
        # One-hot encode the 'year' column
        year_col_name = "year"
        if year_col_name in df_total.columns:
            print(f"df_total b4: {df_total[year_col_name]}")
            one_hot_df = pd.get_dummies(df_total[year_col_name], prefix=year_col_name, dtype=int)
            # one_hot_df.columns = one_hot_df.columns.str.replace('.', '_') # replace "." with "_" in year columns
    
            # Concatenate the new dummy columns with the original DataFrame
            df_total = pd.concat([df_total, one_hot_df], axis=1)
            
            # Drop the original 'Segment' column
            df_total.drop(year_col_name, axis=1, inplace=True)
            print(df_total.columns)
            # print(df_total)
            filtered_df = df_total.filter(like=year_col_name, axis=1) # axis=1 specifies filtering columns
            training_features.remove(year_col_name)
            training_features = training_features + list(one_hot_df.columns)
            print(f"training_features after: {training_features}")
            print(f"df_total after: {filtered_df}")
            print(f"one_hot_df.columns: {one_hot_df.columns}")
    # one hot-encode end ----------------------------------------------------

    
    # apply random shuffle, so that signal and bkg samples get mixed up well
    random_seed_val = 125
    df_total = df_total.sample(frac=1, random_state=random_seed_val)
    # print(f"df_total after shuffle: {df_total}")
    

    
    # print(f"df_total: {df_total}")
    print("starting prepare_dataset")
    df_total = prepare_dataset(df_total, training_samples)
    print("prepare_dataset done")
    print(f"df_total.columns: {df_total.columns}")

    training_features_prepared = prepare_features_from_df(df_total, training_features)
    if not training_features_prepared:
        raise RuntimeError("No training features could be resolved from the training dataframe schema.")
    classifier_train(df_total, args, training_samples, training_features_prepared, random_seed_val, save_path, do_hyperparam_search=do_hyperparam_search, n_trials=sysargs.n_trials)
    # evaluation(df_total, args)
    #df.to_pickle('/depot/cms/hmm/purohita/coffea/eval_dataset.pickle')
    runtime = int(time.time()-start_time)
    print(f"Success! run time is {runtime} seconds")
