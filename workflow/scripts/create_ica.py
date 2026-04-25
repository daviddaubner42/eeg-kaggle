import os
import pickle
import argparse
import mne

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--outdir", type=str)
parser.add_argument("--l_freq", type=float)
parser.add_argument("--subid", type=str)
parser.add_argument("--series", type=str)
args = parser.parse_args()

with open(args.input, "rb") as f:
    raw = pickle.load(f)

filt_raw = raw.filter(l_freq=args.l_freq, h_freq=None)

ica = mne.preprocessing.ICA(n_components=32, max_iter=800)
ica.fit(filt_raw)

fig = ica.plot_sources(raw)

with open(os.path.join(args.outdir, f"sub-{args.subid}_ses-{args.series}_ICA.pkl"), "wb") as f:
    pickle.dump(ica, f)

fig.savefig(os.path.join(args.outdir, f"sub-{args.subid}_ses-{args.series}_ICA_sources.png"))