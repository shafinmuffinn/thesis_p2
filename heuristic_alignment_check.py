import pickle
from pathlib import Path

# Point this at wherever your pickles live (Drive folder on the 4080,
# or a synced copy on the Mac if you have Drive Desktop here)
EAV_PICKLES = Path("G:/My Drive/Thesis_EAV/Input_images")   # adjust

import numpy as np
for sub in [1, 2, 3]:
    print(f"\n=== Subject {sub:02d} ===")
    for mod, suf in [("Audio", "aud"), ("Vision", "vis"), ("EEG", "eeg")]:
        with open(EAV_PICKLES / mod / f"subject_{sub:02d}_{suf}.pkl", "rb") as f:
            tr_x, tr_y, te_x, te_y = pickle.load(f)
        all_y = np.concatenate([np.asarray(tr_y), np.asarray(te_y)])
        counts = np.bincount(all_y, minlength=5)
        print(f"  {mod:6s}: total {len(all_y):4d} trials  "
              f"per-class counts: {counts.tolist()}")
