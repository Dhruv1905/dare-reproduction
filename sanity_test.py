import numpy as np
from src.dare import compute_distance

X_train = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 2.0],
])
x_d = np.array([1.0, 1.0])

d = compute_distance(X_train[0], X_train, x_d)
print("Test 1 (self-distance):", d)
print("  Expected: first entry is 0.0\n")

d1 = compute_distance(np.array([2.0, 0.0]), np.array([[0.0, 0.0]]), np.array([1.0, 1.0]))
d2 = compute_distance(np.array([4.0, 0.0]), np.array([[0.0, 0.0]]), np.array([2.0, 2.0]))
print("Test 2 (scale invariance):", d1, "vs", d2)
print("  Expected: identical values\n")

d1 = compute_distance(np.array([1.0, 0.0]), np.array([[0.0, 0.0]]), np.array([1.0, 1.0]))
d2 = compute_distance(np.array([0.0, 1.0]), np.array([[0.0, 0.0]]), np.array([1.0, 1.0]))
print("Test 3 (feature symmetry):", d1, "vs", d2)
print("  Expected: identical values\n")

close = compute_distance(np.array([1.0, 0.0]), np.array([[0.0, 0.0]]), np.array([10.0, 10.0]))
far   = compute_distance(np.array([1.0, 0.0]), np.array([[0.0, 0.0]]), np.array([0.1, 0.1]))
print("Test 4 (x_d shrinks -> distance grows):", close, "<<", far)
print("  Expected: 'far' is much larger than 'close'")

from src.dare import exponential_kernel

print("\n--- Kernel tests ---\n")

# Test 5: at distance 0, DoC must be exactly 1.
k = exponential_kernel(np.array([0.0]))
print("Test 5 (DoC at zero distance):", k)
print("  Expected: [1.0]\n")

# Test 6: monotonic decay — bigger distance -> smaller DoC.
k = exponential_kernel(np.array([0.0, 0.5, 1.0, 2.0, 5.0]))
print("Test 6 (monotonic decay):", k)
print("  Expected: strictly decreasing, all in [0, 1]\n")

# Test 7: end-to-end — take a real distance from compute_distance
# and turn it into a DoC. The self-distance training point should give DoC=1;
# the farther ones should give DoC in (0, 1).
d = compute_distance(X_train[0], X_train, x_d)
doc = exponential_kernel(d)
print("Test 7 (end-to-end distance -> DoC):", doc)
print("  Expected: first is 1.0, rest are in (0, 1) and decreasing\n")

# Test 8: verify the L=0.2 hook. When distance = |ln(0.2)|/2, DoC should = 0.2 (by design).
# This is the paper's promise: x_d is the tolerance AT which DoC reaches L.
test_dist = np.array([abs(np.log(0.2)) / 2])
print("Test 8 (L=0.2 hook, DoC at the tolerance boundary):", exponential_kernel(test_dist))
print("  Expected: approximately [0.2]")

from src.dare import compute_padoc

print("\n--- PADoC tests ---\n")

# Fresh 2D-input, 1D-target setup.
X_train = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
y_train = np.array([10.0, 20.0, 30.0, 40.0])
x_d_joint = np.array([0.5, 0.5, 10.0])  # tolerances: 0.5 on inputs, 10 on target

# Test 9: predict exactly on a training point, with the correct target.
# PADoC should be highest here (self-vote of 1.0 dominates the N=3 average).
padoc = compute_padoc(
    x_test=np.array([0.0, 0.0]),
    y_test=10.0,
    X_train=X_train, y_train=y_train,
    x_d=x_d_joint, N=3,
)
print(f"Test 9 (on training point, correct prediction): PADoC = {padoc:.4f}")
print("  Expected: high, close to (1.0 + something + something) / 3\n")

# Test 10: same input, but a WRONG prediction (target far from training).
# Same inputs, but y_test = 500 (nowhere near any training y). PADoC should collapse.
padoc = compute_padoc(
    x_test=np.array([0.0, 0.0]),
    y_test=500.0,
    X_train=X_train, y_train=y_train,
    x_d=x_d_joint, N=3,
)
print(f"Test 10 (on training input, WRONG prediction): PADoC = {padoc:.6f}")
print("  Expected: near zero — this is the joint-space check firing\n")

# Test 11: input FAR from any training point (input-space OOD), correct-ish prediction.
padoc = compute_padoc(
    x_test=np.array([100.0, 100.0]),
    y_test=25.0,
    X_train=X_train, y_train=y_train,
    x_d=x_d_joint, N=3,
)
print(f"Test 11 (OOD input, plausible prediction): PADoC = {padoc:.6f}")
print("  Expected: near zero — input-space OOD\n")

# Test 12: single training point, N=3 requested. Should not crash;
# should just average over what's available.
padoc = compute_padoc(
    x_test=np.array([0.0, 0.0]),
    y_test=10.0,
    X_train=np.array([[0.0, 0.0]]),
    y_train=np.array([10.0]),
    x_d=x_d_joint, N=3,
)
print(f"Test 12 (only 1 training point, N=3 requested): PADoC = {padoc:.4f}")
print("  Expected: 1.0 (perfect self-match, no crash)")

from src.dare import accept_reject, dare_batch

print("\n--- Decision + batch tests ---\n")

# Test 13: threshold logic.
# PADoC=0.6 with default xi=0.5 -> accept; PADoC=0.4 -> reject.
print("Test 13 (threshold logic):",
      accept_reject(np.array([0.6, 0.4, 0.5, 0.0, 1.0])))
print("  Expected: [True, False, True, False, True]\n")

# Test 14: Q_X scaling.
# PADoC=0.6, Q_X=0.5 -> r=0.3 < 0.5 -> reject.
# A mediocre model shouldn't be trusted even when data supports it.
print("Test 14 (Q_X=0.5 shrinks a 0.6 PADoC below threshold):",
      accept_reject(0.6, Q_X=0.5))
print("  Expected: False\n")

# Test 15: batch mode end-to-end.
X_train_b = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y_train_b = np.array([10.0, 20.0, 30.0, 40.0])
x_d_b = np.array([0.5, 0.5, 10.0])

# Three test cases: on-training (should accept), OOD input (should reject),
# on-input-but-wrong-prediction (should reject via joint space).
X_test_b = np.array([[0.0, 0.0], [100.0, 100.0], [0.0, 0.0]])
y_pred_b = np.array([10.0, 25.0, 500.0])

padocs, accepted = dare_batch(X_test_b, y_pred_b, X_train_b, y_train_b, x_d_b, N=3)
print(f"Test 15 (batch): padocs = {padocs}")
print(f"                 accepted = {accepted}")
print("  Expected: padocs ~ [0.33, 0.0, 0.0]; accepted = [False, False, False]")
print("  (Note: first PADoC is 0.33 because only 1 of 3 nearest points fully vouches.)\n")

# Test 16: batch with a clearly-in-distribution case.
# Put the test point EXACTLY on a training point with the correct target.
padocs, accepted = dare_batch(
    np.array([[1.0, 1.0]]),          # sits on training point 3
    np.array([40.0]),                 # matching its training target
    X_train_b, y_train_b,
    x_d=np.array([0.5, 0.5, 10.0]), N=3,
)
print(f"Test 16 (well-supported prediction): padoc = {padocs[0]:.4f}, accepted = {accepted[0]}")
print("  Expected: padoc >= 0.33 (one full self-vote); accepted may still be False")
print("  because the other 2 nearest points are 1 unit away (tight tolerance).")

def test_17_adaptive_reduces_to_vanilla():
    """alpha=0 => no widening => identical to vanilla fast version."""
    rng = np.random.default_rng(0)
    X_tr = rng.uniform(0, 1, size=(500, 2))
    y_tr = X_tr.sum(axis=1) + rng.normal(0, 0.01, 500)
    X_te = rng.uniform(0, 1, size=(50, 2))
    y_te = X_te.sum(axis=1)

    from src.dare import dare_batch_fast, dare_batch_adaptive
    pad_v, dec_v = dare_batch_fast(X_te, y_te, X_tr, y_tr, x_d=[0.1, 0.1, 0.1])
    pad_a, dec_a, s = dare_batch_adaptive(X_te, y_te, X_tr, y_tr,
                                          x_d=[0.1, 0.1, 0.1], alpha=0.0)
    assert np.allclose(pad_v, pad_a, atol=1e-10), "alpha=0 should match vanilla"
    assert np.array_equal(dec_v, dec_a)
    assert np.allclose(s, 1.0)
    print("test 17 passed: adaptive with alpha=0 == vanilla")


def test_18_adaptive_widens_in_sparse_region():
    """A test point far from a dense cluster should get s > 1 (capped)."""
    X_tr = np.random.default_rng(1).normal(0, 0.05, size=(500, 2))
    y_tr = np.zeros(500)
    X_te = np.array([[1.0, 1.0]])
    y_te = np.array([0.0])

    from src.dare import dare_batch_adaptive
    _, _, s = dare_batch_adaptive(X_te, y_te, X_tr, y_tr,
                                  x_d=[0.1, 0.1, 0.1],
                                  alpha=0.5, max_scale=2.0)
    assert 1.0 < s[0] <= 2.0, f"expected widening, got s={s[0]}"
    print(f"test 18 passed: sparse point got s={s[0]:.3f}")
    print("\n--- Adaptive tests ---\n")
    
test_17_adaptive_reduces_to_vanilla()
test_18_adaptive_widens_in_sparse_region()