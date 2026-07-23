"""
Combine individual TransientData CSVs from the VAE regenerator into
the four splits used by the paper: dTrain, dTest1, dTest2, dTest3.

Reads from: <VAE_repo>/Generated_Data/TransientData*.csv
Writes to:  data/<split>.csv with columns v_T, T_UP, T_FCL.

Split partitioning is done by transient index (not row-random), so a
whole transient's timeline stays in one split — matching the paper's setup.
"""

import os
import glob
import numpy as np
import pandas as pd

# EDIT THIS PATH if you cloned the VAE repo elsewhere.
VAE_REPO = r"C:\Users\dhruv\Documents\VAE_NAMAC_data"
SRC_DIR = os.path.join(VAE_REPO, "Generated_Data")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

COLUMNS = ["time", "v_T", "T_UP", "T_FCL"]


def load_transient(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df[["v_T", "T_UP", "T_FCL"]]


def build_split(indices, split_name, shift=None):
    """shift : dict like {'T_UP': +20.0, 'T_FCL': +30.0} or None."""
    frames = []
    for i in indices:
        path = os.path.join(SRC_DIR, f"TransientData{i}.csv")
        frames.append(load_transient(path))
    df = pd.concat(frames, ignore_index=True)
    if shift:
        for col, delta in shift.items():
            df[col] = df[col] + delta
    out_path = os.path.join(OUT_DIR, f"{split_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"{split_name}: {len(indices)} transients, {len(df)} rows -> {out_path}"
          + (f"  (shift: {shift})" if shift else ""))


def main():
    # Paper's Table 2: dTrain=125, dTest1/2/3 = 255 each.
    # We have 635 total = 125 + 170 + 170 + 170. Fewer test rows than paper,
    # but proportions and roles are preserved. All transients from the VAE
    # are drawn from the SAME distribution, so we can't literally reproduce
    # dTest2 (hotter) or dTest3 (colder) here — those required different
    # simulator settings the paper had access to. So all three tests
    # are effectively "same distribution as dTrain" here.
    all_ids = list(range(1, 636))
    np.random.seed(0)
    shuffled = np.random.permutation(all_ids)

    dTrain = sorted(shuffled[:125].tolist())
    dTest1 = sorted(shuffled[125:295].tolist())
    dTest2 = sorted(shuffled[295:465].tolist())
    dTest3 = sorted(shuffled[465:635].tolist())

    build_split(dTrain, "dTrain")
    build_split(dTest1, "dTest1")
    build_split(dTest2, "dTest2", shift={"T_UP": +8.0, "T_FCL": +12.0})
    build_split(dTest3, "dTest3", shift={"T_UP": -8.0, "T_FCL": -12.0})


if __name__ == "__main__":
    main()