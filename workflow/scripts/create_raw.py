import os
import pickle
import argparse
import mne

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--inp_dir", type=str)
parser.add_argument("--subid", type=str)
parser.add_argument("--series", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

train_og = pd.read_csv(os.path.join(args.inp_dir, f"subj{args.subid}_series{args.series}_data.csv"))
train_events = pd.read_csv(os.path.join(args.inp_dir, f"subj{args.subid}_series{args.series}_events.csv"))

train = pd.merge(train_og, train_events, on="id")

event_dict = {
    "HandStart": 1,
    "FirstDigitTouch": 2, 
    "BothStartLoadPhase": 3,
    "LiftOff": 4,
    "Replace": 5, 
    "BothReleased": 6
}

train["STIM"] = np.zeros((len(train)))

for event, id in event_dict.items():
    train.loc[train[event] == 1, "STIM"] = id

chnames = list(train_og.columns[1:]) + ["STIM"]
chtypes = ["eeg" for i in chnames]
chtypes[-1] = "stim"
info = mne.create_info(chnames, 500, chtypes)

raw_arr = train[chnames].to_numpy().T * 1e-6
raw_arr[-1] *= 1e6

raw = mne.io.RawArray(raw_arr, info)

with open(args.output, "wb") as f:
    pickle.dump(raw, f)