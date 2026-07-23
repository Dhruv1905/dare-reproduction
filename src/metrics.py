"""
Metrics for evaluating DARE, using the paper's flipped-positive convention:
    "positive" = DARE ACCEPTED the prediction.

- TP: DARE accepted, prediction was actually within tolerance.
- FP: DARE accepted, prediction was actually wrong (the DANGEROUS error).
- TN: DARE rejected, prediction was actually wrong.
- FN: DARE rejected, prediction was actually within tolerance (the WASTEFUL error).

f_peril    = FDR = FP / (FP + TP)   -- Eq. 8
f_degrade  = FNR = FN / (FN + TP)   -- Eq. 7
F1         = 2(1-FDR)(1-FNR)/(2-FDR-FNR) -- Eq. 9
F_max      = F1 if DARE accepted everything (baseline).
"""

import numpy as np


def is_prediction_correct(y_true, y_pred, tolerance):
    """|y_pred - y_true| <= tolerance, elementwise."""
    return np.abs(np.asarray(y_pred) - np.asarray(y_true)) <= tolerance


def confusion_matrix(accepted, correct):
    """Return (TP, FP, TN, FN) under the flipped-positive convention."""
    accepted = np.asarray(accepted, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    TP = int(np.sum(accepted & correct))
    FP = int(np.sum(accepted & ~correct))
    TN = int(np.sum(~accepted & ~correct))
    FN = int(np.sum(~accepted & correct))
    return TP, FP, TN, FN


def f_peril(TP, FP):
    """Eq. 8. Fraction of accepted predictions that are actually wrong."""
    if TP + FP == 0:
        return 0.0
    return FP / (FP + TP)


def f_degrade(TP, FN):
    """Eq. 7. Fraction of correct predictions that DARE wastefully rejected."""
    if TP + FN == 0:
        return 0.0
    return FN / (FN + TP)


def f1_score(TP, FP, FN):
    """Eq. 9, expressed in TP/FP/FN form (matches paper).
    Returns 0.0 when nothing was accepted (TP+FP == 0) — otherwise F1
    spuriously reports 1.0 for total-rejection cases."""
    if TP + FP == 0:
        return 0.0
    fdr = f_peril(TP, FP)
    fnr = f_degrade(TP, FN)
    denom = 2 - fdr - fnr
    if denom == 0:
        return 0.0
    return 2 * (1 - fdr) * (1 - fnr) / denom


def f_max(y_true, y_pred, tolerance):
    """Baseline: F1 if DARE accepted EVERY prediction. No filtering."""
    correct = is_prediction_correct(y_true, y_pred, tolerance)
    accepted = np.ones_like(correct, dtype=bool)
    TP, FP, TN, FN = confusion_matrix(accepted, correct)
    return f1_score(TP, FP, FN)


def evaluate_dare(y_true, y_pred, accepted, tolerance):
    """
    One-call summary: returns dict with all four metrics + counts.
    Use this in the validation notebook.
    """
    correct = is_prediction_correct(y_true, y_pred, tolerance)
    TP, FP, TN, FN = confusion_matrix(accepted, correct)
    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "total": TP + FP + TN + FN,
        "f_peril": f_peril(TP, FP),
        "f_degrade": f_degrade(TP, FN),
        "F1": f1_score(TP, FP, FN),
        "F_max": f_max(y_true, y_pred, tolerance),
    }   