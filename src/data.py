"""Load the VAE-generated EBR-II loss-of-flow datasets from data/."""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FEATURES = ["v_T", "T_UP"]
TARGET = "T_FCL"


def load_split(name):
    """
    name : 'dTrain' | 'dTest1' | 'dTest2' | 'dTest3'
    Returns X (n, 2), y (n,)
    """
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    return X, y


def load_all():
    return {name: load_split(name) for name in ["dTrain", "dTest1", "dTest2", "dTest3"]}