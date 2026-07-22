"""
DARE core: Data Auditing for Reliability Evaluation.

Reproduces the method from Chen, Bao, Dinh (2024),
Reliability Engineering & System Safety 250, 110266.
"""

import numpy as np


def compute_distance(x_test, X_train, x_d, L=0.2):
    """
    Weighted distance from one test point to every training point.
    Implements Eq. 2 of the paper, with beta derived from Eq. 5.

    Parameters
    ----------
    x_test : np.ndarray, shape (d,)
        A single test point in the joint (input, prediction) space.
    X_train : np.ndarray, shape (n, d)
        Training points in the same joint space.
    x_d : np.ndarray, shape (d,)
        Interpolation length per feature (the main user-set knob).
        One entry per input feature PLUS one per target.
    L : float, default 0.2
        The DoC level at which x_d applies. Paper's default is 0.2.
        Do not set to 0 — it's an asymptote of the kernel.

    Returns
    -------
    distances : np.ndarray, shape (n,)
        Weighted distance from x_test to each training point.
    """
    x_test = np.asarray(x_test, dtype=float)
    X_train = np.asarray(X_train, dtype=float)
    x_d = np.asarray(x_d, dtype=float)

    # Eq. 5: convert interpolation length x_d to decay rate beta, per feature.
    # beta is the DIAGONAL of V. We never form V or V^-1 explicitly.
    beta = (x_d ** 2) / (np.log(L) ** 2)   # shape (d,)

    # Element-wise (x' - x_i) for every training point i.
    # Broadcasting: (d,) - (n, d) -> (n, d)
    diff = x_test - X_train                # shape (n, d)

    # Eq. 2 unrolled for a diagonal V:
    # (x' - x)^T V^-1 (x' - x) = sum over features of (diff_j)^2 / beta_j
    # We divide by beta instead of multiplying by V^-1 — same thing, faster.
    weighted_sq = (diff ** 2) / beta       # shape (n, d)
    distances = np.sqrt(weighted_sq.sum(axis=1))   # shape (n,)

    return distances

def exponential_kernel(distances):
    """
    Convert weighted distances to Degree of Congruency (DoC) via Eq. 1.

    DoC = 1 means the test point sits on top of a training point
    (perfect support). DoC → 0 means the training point is too far
    away to vouch for the test point.

    Parameters
    ----------
    distances : np.ndarray, shape (n,)
        Weighted distances from compute_distance().

    Returns
    -------
    doc : np.ndarray, shape (n,)
        Degree of Congruency for each training point, in [0, 1].
    """
    distances = np.asarray(distances, dtype=float)
    return np.exp(-2.0 * distances)

def compute_padoc(x_test, y_test, X_train, y_train, x_d, N=3, L=0.2):
    """
    Proximity Averaged Degree of Congruency for a single test prediction.
    Implements Eq. 6, which evaluates the kernel on the JOINT
    (input, prediction) space — not just inputs.

    Parameters
    ----------
    x_test : np.ndarray, shape (d_in,)
        Input features of the test point.
    y_test : float or np.ndarray shape (d_out,)
        The MODEL'S PREDICTION on x_test — not the ground truth.
        DARE evaluates predictions against training evidence.
    X_train : np.ndarray, shape (n, d_in)
        Training input features.
    y_train : np.ndarray, shape (n,) or (n, d_out)
        Training targets (the true values seen during training).
    x_d : np.ndarray, shape (d_in + d_out,)
        Interpolation length per feature. Length = inputs + outputs.
        This is the point where the joint-space assumption matters:
        the caller must supply x_d entries for BOTH inputs and target.
    N : int, default 3
        Number of nearest training points to average over.
    L : float, default 0.2
        DoC level at which x_d applies. Paper default.

    Returns
    -------
    padoc : float
        PADoC score in [0, 1]. Higher = more training support for
        the prediction. Multiply by Q_X to get instantaneous success rate.
    """
    # Coerce y_test and y_train to 2D so concatenation works uniformly
    # whether the target is a scalar or a vector.
    y_test = np.atleast_1d(y_test).astype(float)             # (d_out,)
    y_train = np.asarray(y_train, dtype=float)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)                     # (n, 1)

    X_train = np.asarray(X_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)

    # THE joint-space step: concatenate inputs and target into one vector.
    # This is what makes DARE check "prediction agrees with training", not
    # just "inputs look familiar".
    joint_test = np.concatenate([x_test, y_test])            # (d_in + d_out,)
    joint_train = np.concatenate([X_train, y_train], axis=1) # (n, d_in + d_out)

    # Weighted distance in the joint space.
    distances = compute_distance(joint_test, joint_train, x_d, L=L)

    # DoC per training point.
    doc = exponential_kernel(distances)                      # (n,)

    # Top-N by DoC (equivalently: N smallest distances).
    # If we have fewer than N training points, use all of them.
    n_available = len(doc)
    n_use = min(N, n_available)
    top_n = np.sort(doc)[-n_use:]                            # ascending, take last N

    # Average — this is μ, the PADoC.
    padoc = top_n.mean()

    return float(padoc)

def accept_reject(padoc, Q_X=1.0, xi=0.5):
    """
    Apply the DARE decision rule (paper §3.2).

    r = Q_X * padoc  -> accept if r >= xi, else reject.

    Parameters
    ----------
    padoc : float or np.ndarray
        PADoC score(s) from compute_padoc.
    Q_X : float, default 1.0
        Model training performance. Paper uses 1.0 throughout Section 4
        to isolate DARE's effect. Real deployment would use the model's
        training F1 / accuracy.
    xi : float, default 0.5
        Acceptance threshold. Paper default.

    Returns
    -------
    accepted : bool or np.ndarray of bool
        True = DARE trusts this prediction, False = reject.
    """
    r = Q_X * np.asarray(padoc)
    return r >= xi


def dare_batch(X_test, y_pred, X_train, y_train, x_d, N=3, Q_X=1.0, xi=0.5, L=0.2):
    """
    Run DARE over a whole test set.

    Parameters
    ----------
    X_test : np.ndarray, shape (m, d_in)
        Test inputs.
    y_pred : np.ndarray, shape (m,) or (m, d_out)
        MODEL PREDICTIONS on X_test (not ground truth).
    X_train, y_train : as in compute_padoc.
    x_d : np.ndarray, shape (d_in + d_out,)
        Joint-space interpolation lengths.
    N, Q_X, xi, L : DARE hyperparameters.

    Returns
    -------
    padocs : np.ndarray, shape (m,)
        PADoC score for each test sample.
    accepted : np.ndarray of bool, shape (m,)
        DARE's accept/reject decision for each test sample.
    """
    X_test = np.asarray(X_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim == 1:
        y_pred_iter = y_pred.reshape(-1, 1)
    else:
        y_pred_iter = y_pred

    m = X_test.shape[0]
    padocs = np.empty(m)

    for i in range(m):
        padocs[i] = compute_padoc(
            x_test=X_test[i],
            y_test=y_pred_iter[i],
            X_train=X_train,
            y_train=y_train,
            x_d=x_d,
            N=N,
            L=L,
        )

    accepted = accept_reject(padocs, Q_X=Q_X, xi=xi)
    return padocs, accepted

from sklearn.neighbors import BallTree


def dare_batch_fast(X_test, y_pred, X_train, y_train, x_d, N=3, Q_X=1.0, xi=0.5, L=0.2):
    """
    Fast DARE using a BallTree in the pre-whitened joint space.

    Trick: instead of computing weighted distance point-by-point, we
    rescale every coordinate by sqrt(beta) once so raw Euclidean distance
    in that space == the paper's weighted distance. Then a BallTree
    gives us the N nearest neighbors in O(log n) per query.

    Returns (padocs, accepted). Same signature as dare_batch.
    """
    X_test = np.asarray(X_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    x_d = np.asarray(x_d, dtype=float)

    # Build joint spaces.
    joint_test = np.concatenate([X_test, y_pred], axis=1)
    joint_train = np.concatenate([X_train, y_train], axis=1)

    # Pre-whitening: divide each coordinate by sqrt(beta).
    # Then Euclidean distance in whitened space == paper's weighted distance.
    beta = (x_d ** 2) / (np.log(L) ** 2)
    scale = np.sqrt(beta)                    # shape (d,)
    joint_train_w = joint_train / scale
    joint_test_w = joint_test / scale

    # BallTree lookup: find N nearest training points per test point.
    tree = BallTree(joint_train_w, leaf_size=40)
    n_use = min(N, len(joint_train_w))
    dists, _ = tree.query(joint_test_w, k=n_use)   # shape (m, n_use)

    # DoC per neighbor, then average.
    doc = np.exp(-2.0 * dists)                     # shape (m, n_use)
    padocs = doc.mean(axis=1)                      # shape (m,)

    accepted = (Q_X * padocs) >= xi
    return padocs, accepted