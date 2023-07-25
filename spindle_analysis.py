import mne
from mne.time_frequency import psd_array_multitaper

from itertools import combinations
from scipy.stats import ttest_rel
import scipy
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy.interpolate import interp1d

from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score

from functools import partial
from pathlib import Path

trim_mean = partial(scipy.stats.trim_mean, proportiontocut=0.1)


def get_spindles_annotations(raw, l_freq=10, h_freq=15, win_width=0.75,
                             ch_name=None, low_th=1.2, high_th=3.5,
                             min_duration=0.5, max_duration=10.0,
                             robust=False):
    resamp_freq = 50
    
    # 10-15 Hz bandpass
    if ch_name is None:
        ch_name = [ch_name for ch_name in raw.ch_names
                   if "eeg" in ch_name.lower() or "channel" in ch_name.lower()][0]
    bp_data = raw.copy().pick(ch_name).load_data().filter(l_freq, h_freq)
    bp_data = bp_data.resample(resamp_freq)
    times = bp_data.times
    bp_data = bp_data.get_data().squeeze()

    series = pd.Series(bp_data**2, index=times)

    # Rolling windows for MS computation
    rolling = series.rolling(int(win_width*resamp_freq), 1, True)
    if robust:
        res = np.sqrt(rolling.agg(trim_mean))**3  # cubed RMS
    else:
        res = np.sqrt(rolling.mean())**3  # cubed RMS        

    # Rolling window for computing the average MS signal amplitude
    page_width = 10*60
    rolling = series.rolling(int(page_width*resamp_freq), 1, True)
    if robust:
        res_page = np.sqrt(rolling.agg(trim_mean))**3  # cubed RMS
    else:
        res_page = np.sqrt(rolling.mean())**3  # cubed RMS        

    # Thresholding
    df = pd.DataFrame({"high": res > high_th*res_page,
                       "low": res > low_th*res_page})

    # Making windows based on the low threshold
    mask = df.low.diff().fillna(False)

    # Force transitions if starts or ends with True values
    mask.iloc[0] = df.low.values[0]
    mask.iloc[-1] = df.low.values[-1]

    tmp = df.index[mask]
    starts = tmp[::2]
    stops = tmp[1::2]

    # Selecting only the windows with a peak above the high threshold
    tmp = np.array([(start, stop) for start, stop in zip(starts, stops)
                    if np.sum(df.loc[start:stop, "high"])])
    if not len(tmp):
        return mne.Annotations([], [], [], orig_time=raw.info["meas_date"])

    # Checking window duration based on the low threshold
    durations = np.diff(tmp).squeeze()
    valid_windows = tmp[(durations >= min_duration) &
                        (durations <= max_duration), :]
    if not len(valid_windows):
        return mne.Annotations([], [], [], orig_time=raw.info["meas_date"])

    assert np.all(np.diff(valid_windows).squeeze() >= min_duration)

    return mne.Annotations(valid_windows[:, 0],
                           np.diff(valid_windows).squeeze(),
                           "spindle", orig_time=raw.info["meas_date"])


def get_manual_annotations(file_path, raw):

    spin_day1 = pd.read_excel(file_path)
    spin_day1 = spin_day1.drop(0, axis=0)
    spin_day1 = spin_day1[['Start Time', 'End Time']]

    spin_day1.columns = ["start", "stop"]

    spin_day1["meas_date"] = raw.info["meas_date"]
    spin_day1.start = spin_day1.start.astype(spin_day1.meas_date.dtype)
    spin_day1.stop = spin_day1.stop.astype(spin_day1.meas_date.dtype)

    starts = (spin_day1.start-spin_day1.meas_date)
    starts = starts.astype('timedelta64[ms]')/1000
    stops = (spin_day1.stop-spin_day1.meas_date)
    stops = stops.astype('timedelta64[ms]')/1000

    return mne.Annotations(starts, stops-starts, "spindle_manual",
                           orig_time=raw.info["meas_date"])


def fix_edf(stages_df):
    vals, counts = np.unique(stages_df.start, return_counts=True)
    if len(np.unique(counts)) == 1 and np.unique(counts)[0] == 1:
        return stages_df  # nothing to fix
    for val in vals[np.where(counts != 6)]:
        stages_df = stages_df[stages_df.start != val]    
    for i in range(1, 6):
        assert np.all(stages_df.iloc[::6]["start"].values == 
                      stages_df.iloc[i::6]["start"].values)
        stages_df.iloc[i::6].loc[:, "start"] += 10*i
    return stages_df


def get_sleep_stages_df(file_path, raw, stage_duration=10.0):
    if file_path.suffix == ".csv":
        stages = pd.read_csv(file_path)
    else:
        stages = pd.read_excel(file_path)

    stages = stages.drop(index=0)[["Rodent Sleep" if "Rodent Sleep" in stages else "Label", 
                                   "Start Time" if "Start Time" in stages else "Time Stamp"]]

    stages.columns = ["label", "start"]
    starts = stages.start.astype('datetime64[ns, UTC]')
    stages.start = (starts - raw.info["meas_date"]).astype('timedelta64[s]')
    if file_path.suffix == ".csv":
        stages = fix_edf(stages)
    return stages


def get_sleep_stages(file_path, raw, stage_duration=10.0):
    stages = get_sleep_stages_df(file_path, raw, stage_duration)

    return np.sum([mne.Annotations(stages.start[stages.label == stage],
                                   stage_duration,
                                   f"stage_{stage}",
                                   orig_time=raw.info["meas_date"])
                   for stage in stages.label.unique()])


def filter_annotations(annotations, tmin, tmax):
    # keep annotation which middle point is within tmin and tmax, without
    # croping them if they party overlap outside of this range
    mid_points = [annot["onset"] + annot["duration"]/2
                  for annot in annotations]
    dat = np.array([(annot["onset"], annot["duration"], annot["description"])
                    for annot, mid_point in zip(annotations, mid_points)
                    if tmin < mid_point < tmax]).T
    if len(dat):
        onset, duration, description = dat
        return mne.Annotations(onset, duration, description,
                               orig_time=annotations.orig_time)
    return mne.Annotations([], [], [], orig_time=annotations.orig_time)


def filter_annots_per_stage(annot_to_filter, annot_stages, stage_in="stage_S"):
    if not len(annot_to_filter):
        return annot_to_filter
    annots = {"onset": [annot["onset"] for annot in annot_stages],
              "duration": [annot["duration"] for annot in annot_stages],
              "description": [annot["description"] for annot in annot_stages]}
    annots = pd.DataFrame(annots)

    cond = annots.description == stage_in
    discontinuities = np.where(np.diff(annots[cond].onset) != 10)[0]+1
    discontinuities = np.concatenate([[0], discontinuities, [np.sum(cond)]])
    block_lenghts = np.diff(discontinuities)*10
    block_starts = annots[cond].onset.values[discontinuities[:-1]]

    lst_annots = [filter_annotations(annot_to_filter, start, start+duration)
                  for start, duration in zip(block_starts, block_lenghts)]
    annotations = np.sum(lst_annots)
    if not isinstance(annotations, mne.Annotations):
        # Happens when there is not annotations
        return lst_annots[0]
    return annotations


def file_list_to_dict(files):
    keys = [(path.parent.name, path.name.split(".")[-2].split("_")[-1])
            for path in files]
    return dict(zip(keys, files))


def get_annotated_raw(edf_file, spindle_file, sleep_file,
                      ch_name=None):
    raw = mne.io.read_raw_edf(edf_file)

    spin_annotations = get_spindles_annotations(raw, ch_name=ch_name)
    annotations_manual = get_manual_annotations(spindle_file, raw)
    annotations_stages = get_sleep_stages(sleep_file, raw)

    spin_annotations = filter_annots_per_stage(spin_annotations,
                                               annotations_stages)

    print([raw.annotations, spin_annotations,
           annotations_manual, annotations_stages])
    raw.set_annotations(raw.annotations + spin_annotations +
                        annotations_manual + annotations_stages)
    return raw


def get_sample_df(raw, sleep_file):
    sample_df = pd.DataFrame({"time": raw.times[::10]})
    sample_df["spindle"] = False
    sample_df["spindle_manual"] = False
    sample_df = sample_df.set_index("time")

    annot_df = raw.annotations.to_data_frame()
    starts = annot_df.onset.astype('datetime64[ns, UTC]')
    annot_df["start"] = (starts-raw.info["meas_date"])
    annot_df["start"] = annot_df["start"].astype('timedelta64[ms]')/1000
    annot_df["stop"] = annot_df["start"] + annot_df.duration

    for _, row in annot_df.iterrows():
        sample_df.loc[row.start:row.stop, row.description] = True

    stages = pd.read_excel(sleep_file)
    stages = stages.drop(index=0)[["Label", "Start Time"]]
    stages.columns = ["label", "start"]
    starts = stages.start.astype('datetime64[ns, UTC]')
    stages.start = (starts-raw.info["meas_date"]).astype('timedelta64[s]')
    stage_duration = 10.0

    sample_df["nrem"] = False
    sample_df["rem"] = False
    for _, row in stages.iterrows():
        if row.label == "S":
            sample_df.loc[row.start:row.start+stage_duration, "nrem"] = True
        if row.label == "P":
            sample_df.loc[row.start:row.start+stage_duration, "rem"] = True

    return sample_df


def get_performances(sample_df):
    return {"accuracy": accuracy_score(sample_df.spindle_manual,
                                       sample_df.spindle),
            "f1": f1_score(sample_df.spindle_manual,
                           sample_df.spindle),
            "precision": precision_score(sample_df.spindle_manual,
                                         sample_df.spindle),
            "recall": recall_score(sample_df.spindle_manual,
                                   sample_df.spindle)}


def get_spin_features(raw, event_name, l_freq=10,
                      h_freq=15, ch_name="EEG 1"):
    bp_data = raw.copy().pick(ch_name).load_data().filter(l_freq, h_freq)

    target_freqs = np.arange(10, 15.1, 0.1)
    p2p = []
    powers = []
    for annot in tqdm(raw.annotations):
        if annot["description"] == event_name:
            dat = bp_data.get_data(tmin=annot["onset"],
                                   tmax=annot["onset"]+annot["duration"])
            p2p.append(dat.max()-dat.min())
            power, freqs = psd_array_multitaper(dat, raw.info["sfreq"],
                                                fmin=5, fmax=20, verbose=False)
            try:
                f = interp1d(freqs, power, kind='cubic')
                powers.append(f(target_freqs).squeeze())
            except ValueError:
                powers.append(target_freqs*np.nan)

    spin_feat = pd.DataFrame([(annot["onset"], annot["duration"])
                              for annot in raw.annotations
                              if annot["description"] == event_name],
                             columns=["onset", "duration"])

    # The two systems recorded in different units
    if np.mean(p2p) > .005:
        spin_feat["p2p_amp"] = np.array(p2p)*1e3  # In uV
    elif np.mean(p2p) < .001:
        spin_feat["p2p_amp"] = np.array(p2p)*1e6  # In uV
    elif len(p2p) == 0:
        spin_feat["p2p_amp"] = []
    else:
        raise ValueError(f"Check units. Mean p2p == {np.mean(p2p)}; p2p={p2p}")

    for freq, power_values in zip(target_freqs, np.array(powers).T):
        spin_feat[str(np.round(freq, 1))] = power_values

    spin_feat["event"] = event_name

    return spin_feat


def get_spin_features_events(raw, ch_name, event_names):
    return pd.concat([get_spin_features(raw, event_name, ch_name=ch_name)
                      for event_name in event_names])


def get_plotting_data(raw, ch_name, nb_hours=4, bins_per_hours=5,
                      event_kinds=("spindle", "spindle_manual")):
    feat = get_spin_features_events(raw, ch_name, event_kinds)

    bins = np.linspace(0, nb_hours*60*60, nb_hours*bins_per_hours+1)
    bin_labels = (bins[1:] + bins[:-1])/2/60/60
    feat["bins"] = pd.cut(feat["onset"], bins=bins,
                          labels=bin_labels, right=False)

    target_freqs = feat.loc[:, "10.0":"15.0"].columns
    groups = feat.drop(columns=target_freqs)\
                 .melt(id_vars=["onset", "event", "bins"])\
                 .groupby(["event", "bins", "variable"])
    data = groups.mean().drop(columns="onset")
    data_count = groups.count().drop(columns="onset").reset_index()
    data_count = data_count[data_count.variable == "duration"]
    data_count["variable"] = "count"

    sleep_durations = {}
    for tmin, tmax, bin in zip(bins[:-1], bins[1:], bin_labels):
        loop_annots = raw.annotations.copy().crop(tmin=tmin, tmax=tmax,
                                                  use_orig_time=False)
        sleep_durations[bin] = np.sum([annot["duration"]
                                       for annot in loop_annots
                                       if annot["description"] == "stage_S"])
    sleep_durations = pd.Series(sleep_durations).reset_index()
    sleep_durations.columns = ["bins", "sleep_duration"]

    sleep_durations.bins = pd.Categorical(sleep_durations.bins.astype(float))

    data_density = data_count.copy().merge(sleep_durations, on="bins")
    data_density["value"] /= data_density.sleep_duration/60
    data_density = data_density.drop(columns="sleep_duration")
    data_density["variable"] = "density"

    data = pd.concat([data,
                      data_count.set_index(["event", "bins", "variable"]),
                      data_density.set_index(["event", "bins", "variable"])])
    data = data.sort_values(['event', 'variable', 'bins'])
    return data.reset_index(), feat.drop(columns=["bins"]).reset_index()


def get_sleep_duration_df(sleep_files):
    sleep_duration = {key: np.sum(pd.read_excel(path).iloc[1:, 0] == "S")/6 for key, path in sleep_files.items()}
    sleep_duration = pd.Series(sleep_duration).reset_index()
    sleep_duration.columns = ["animal", "condition", "nrem_min"]
    #sleep_duration.loc[sleep_duration.condition == "Vehicle 1", "condition"] = "Vehicle"
    #sleep_duration.loc[sleep_duration.condition == "Vehicle 2", "condition"] = "Vehicle"
    sleep_duration.loc[sleep_duration.condition == "Intermittent 1", "condition"] = "Intermittent"
    sleep_duration.loc[sleep_duration.condition == "Intermittent 2", "condition"] = "Intermittent"
    sleep_duration = sleep_duration[sleep_duration.condition != "Intermittent"]
    return sleep_duration.set_index(["animal", "condition"])    


def get_pvalues(df, sleep_duration, event_kinds=("spindle",)):
    res_inds = get_stats_ind(df, sleep_duration, event_kinds)

    # Averaging results across Vehicle 1 and Vehicle 2
    res_inds.loc[res_inds.condition == "Vehicle 1", "condition"] = "Vehicle"
    res_inds.loc[res_inds.condition == "Vehicle 2", "condition"] = "Vehicle"
    res_inds = res_inds.groupby(["feature", "condition", "scoring", "animal"]).mean().reset_index()

    p_val_res = []
    for (scoring, feature), df in res_inds.groupby(['scoring', 'feature']):
        for x, y in list(combinations(df.condition.unique(), 2)):
            res = {"scoring": scoring, "feature": feature}
            tmp = df.set_index("animal")  # Aligned by the index when concatenating
            tmp = pd.concat([tmp[tmp.condition == x].rename(columns={"value": x})[x],
                             tmp[tmp.condition == y].rename(columns={"value": y})[y]], 
                            axis=1).dropna()
            res["t-value"], res["p-value"] = ttest_rel(tmp[x], tmp[y])
            res["N"] = len(tmp[x])
            res["condition1"] = x
            res["condition2"] = y
            p_val_res.append(res)

    return pd.DataFrame(p_val_res).sort_values(["scoring", "feature"])


def get_stats_ind(df, sleep_duration, event_kinds=("spindle", "spindle_manual")):
    all_res = []
    for scoring in event_kinds:
        for feature in ["duration", "p2p_amp"]:
            tmp = df.groupby(["event", "animal", "condition"]).mean()[[feature]].reset_index()
            tmp = tmp.pivot_table(index=["event", "animal"], columns="condition")[feature]
            if feature == "p2p_amp":
                tmp.loc[tmp[df.condition.unique()[0]] > 5000, df.condition.unique()] /= 1000

            res = tmp.loc[scoring]
            res["scoring"] = scoring
            res["feature"] = feature
            all_res.append(res)

        feature = "duration"
        tmp = df.groupby(["event", "animal", "condition"]).count()[[feature]].reset_index("event")
        tmp = tmp.merge(sleep_duration, left_index=True, right_index=True)
        tmp[feature] /= tmp.nrem_min
        tmp = tmp.drop(columns="nrem_min").rename(columns={"duration": "density"}).reset_index()
        feature = "density"
        tmp = tmp.pivot_table(index=["event", "animal"], columns="condition")[feature]

        res = tmp.loc[scoring]
        res["scoring"] = scoring
        res["feature"] = feature
        all_res.append(res)

    stats_df = pd.concat(all_res).reset_index().melt(id_vars=["animal", "scoring", "feature"])
    return stats_df.sort_values(["scoring", "feature", "condition"])


def get_stats(df, sleep_duration, event_kinds=("spindle", "spindle_manual")):
    res_inds = get_stats_ind(df, sleep_duration, event_kinds)

    # Averaging results across Vehicle 1 and Vehicle 2
    res_inds.loc[res_inds.condition == "Vehicle 1", "condition"] = "Vehicle"
    res_inds.loc[res_inds.condition == "Vehicle 2", "condition"] = "Vehicle"
    res_inds = res_inds.groupby(["feature", "condition", "scoring", "animal"]).mean().reset_index()

    return pd.concat([res_inds.groupby(["feature", "condition", "scoring"]).mean().rename(columns={"value": "mean"})["mean"],
                      res_inds.groupby(["feature", "condition", "scoring"]).std().rename(columns={"value": "std"})["std"],
                      res_inds.groupby(["feature", "condition", "scoring"]).count().rename(columns={"value": "count"})["count"]],
                     axis=1)


def get_spin_feat_df(df_perf, file_pattern="{animal}_{condition}_spin_feat.csv"):
    dfs = []
    for animal, condition in df_perf[["animal", "condition"]].values:
        path = Path(".") / file_pattern.format(animal=animal,
                                               condition=condition)
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        df['animal'], df['condition'] = path.name.split("_")[:2]
        dfs.append(df)

    df = pd.concat(dfs)
    #df.loc[df.condition == "Vehicle 1", "condition"] = "Vehicle"
    #df.loc[df.condition == "Vehicle 2", "condition"] = "Vehicle"
    df.loc[df.condition == "Intermittent 1", "condition"] = "Intermittent"
    df.loc[df.condition == "Intermittent 2", "condition"] = "Intermittent"
    return df[df.condition != "Intermittent"]


def get_perf_df(study_no, edf_files, sleep_files, 
                spindle_files, recompute=False,
                file_pattern="detection_performance_study{study_no}.csv"):
    file_name = Path(file_pattern.format(study_no=study_no))
    if recompute or not file_name.exists():
        results = []
        for key in tqdm(list(edf_files)):
            raw = mne.io.read_raw_edf(edf_files[key])
            ch_names = raw.pick("eeg").ch_names
            ch_names = [ch_name for ch_name in ch_names 
                        if "eeg" in ch_name.lower() or "channel" in ch_name.lower()]
            for ch_name in ch_names:
                raw = get_annotated_raw(edf_files[key], spindle_files[key], 
                                        sleep_files[key], ch_name)
                sample_df = get_sample_df(raw, sleep_files[key])
                res = get_performances(sample_df)
                res["animal"] = key[0]
                res["condition"] = key[1]
                res["ch_name"] = ch_name
                results.append(res)

        df_perf = pd.DataFrame(results)        
        df_perf = df_perf.sort_values('f1', ascending=False).drop_duplicates(['animal','condition'])
        df_perf = df_perf.reset_index(drop=True).sort_values("animal")
        df_perf.to_csv(file_name)
        return df_perf.drop(columns="ch_name")

    df_perf = pd.read_csv(file_name, index_col=0).reset_index(drop=True)
    return df_perf.sort_values("animal")


def get_whole_raw(edf_file, condition, animal, start_times, ch_name):
    raw = mne.io.read_raw_edf(edf_file)
    tmin = (start_times[(animal, condition)] - start_times[(animal, "whole")]).days*24*60*60 
    tmin += (start_times[(animal, condition)] - start_times[(animal, "whole")]).seconds 
    tmin +=  (start_times[(animal, condition)] - start_times[(animal, "whole")]).microseconds/10e6
    tmax = tmin + 12*60*60
    cropped_raw = raw.copy().crop(tmin, tmax)
    cropped_raw.set_meas_date(start_times[(animal, condition)])
    cropped_raw._cropped_samp = 0      
    return cropped_raw


def get_start_times(whole_edf_files, edf_files):
    start_times = {}
    for animal in tqdm(whole_edf_files):
        for condition in ["baseline", "kynurenine"]:
            raw = mne.io.read_raw_edf(edf_files[(animal, condition)])
            start_times[(animal, condition)] = raw.info["meas_date"]

        raw = mne.io.read_raw_edf(whole_edf_files[animal])    
        start_times[(animal, "whole")] = raw.info["meas_date"]

    return start_times
