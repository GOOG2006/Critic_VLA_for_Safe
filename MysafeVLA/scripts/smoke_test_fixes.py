"""Standalone smoke test for the QP-status and label-leakage fixes.

Does NOT require libero / openpi / transformers — only numpy + cvxpy — so it
runs in Evo1 without triggering the tensorflow-numpy2 import bomb.
"""
import sys
import numpy as np
import cvxpy as cp


# ----------------------------------------------------------------------------
# TEST 1: QP infeasibility is detected (previously silently swallowed)
# ----------------------------------------------------------------------------
def test_qp_infeasible_detected():
    # Contradictory linear constraints -> infeasible
    u = cp.Variable(9)
    c = np.ones(9)
    constraints = [c @ u <= -1.0, c @ u >= 1.0]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(u)), constraints)

    n_qp_infeasible = 0
    qp_ok = False
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if u.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            qp_ok = True
        else:
            print(f"  [test1] caught infeasible: status={prob.status} (u.value is {'None' if u.value is None else 'not None'})")
    except Exception as e:
        print(f"  [test1] caught solver error: {type(e).__name__}: {e}")
    if not qp_ok:
        n_qp_infeasible += 1

    assert not qp_ok, "test1 FAILED: qp_ok=True for infeasible problem"
    assert n_qp_infeasible == 1, f"test1 FAILED: counter={n_qp_infeasible}"
    print("  [test1] PASS: infeasible QP detected, counter incremented")


# ----------------------------------------------------------------------------
# TEST 2: Feasible QP still works
# ----------------------------------------------------------------------------
def test_qp_feasible_ok():
    u = cp.Variable(9)
    c = np.ones(9)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(u)), [c @ u <= 1.0])
    qp_ok = False
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if u.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            qp_ok = True
    except Exception:
        pass
    assert qp_ok, f"test2 FAILED: feasible QP not marked ok, status={prob.status}"
    print(f"  [test2] PASS: feasible QP -> status={prob.status}")


# ----------------------------------------------------------------------------
# TEST 3: Label leakage fix — tail-truncation + collision propagation
# ----------------------------------------------------------------------------
def test_label_leakage_truncation():
    LOOKAHEAD = 10
    # Fake trajectory: 30 steps, collision at step 20, h monotonically decreasing
    trajectory = [{"step": i, "h": 5.0 - 0.1 * i, "collided": (i == 20)} for i in range(30)]

    if len(trajectory) > LOOKAHEAD:
        usable = trajectory[:-LOOKAHEAD]
        for i, t in enumerate(usable):
            future = trajectory[i:i + LOOKAHEAD]
            assert len(future) == LOOKAHEAD
            t["future_collision"] = any(f["collided"] for f in future)
            t["h_future_min"] = min(f["h"] for f in future)

    assert len(usable) == 20, f"len(usable)={len(usable)}, expected 20"
    # Steps 0..10 cannot see step 20 in their 10-step lookahead (their window ends at step i+9)
    #   i=10 -> window 10..19 -> no collision
    #   i=11 -> window 11..20 -> collision
    for i, t in enumerate(usable):
        if i <= 10:
            assert not t["future_collision"], f"step {i} should NOT predict collision (window {i}..{i+LOOKAHEAD-1})"
        else:
            assert t["future_collision"], f"step {i} should predict collision (window {i}..{i+LOOKAHEAD-1})"

    # h_future_min should be strictly < current h for all usable steps (monotonic)
    for i, t in enumerate(usable):
        future_hs = [trajectory[j]["h"] for j in range(i, i + LOOKAHEAD)]
        assert abs(t["h_future_min"] - min(future_hs)) < 1e-9, "h_future_min wrong"
        # Real future-min must be at the end of the window (monotone decrease)
        assert abs(t["h_future_min"] - (5.0 - 0.1 * (i + LOOKAHEAD - 1))) < 1e-9

    print(f"  [test3] PASS: {len(usable)} usable / {len(trajectory)} total, "
          f"tail {len(trajectory) - len(usable)} dropped")


# ----------------------------------------------------------------------------
# TEST 4: Short trajectory (no usable samples)
# ----------------------------------------------------------------------------
def test_label_leakage_short():
    LOOKAHEAD = 10
    trajectory = [{"step": i, "h": 1.0, "collided": False} for i in range(5)]
    if len(trajectory) > LOOKAHEAD:
        usable = trajectory[:-LOOKAHEAD]
    else:
        usable = []
    assert usable == [], "short trajectory should produce zero usable samples"
    print(f"  [test4] PASS: trajectory shorter than LOOKAHEAD -> 0 usable samples")


if __name__ == "__main__":
    print("=" * 60)
    print("Smoke-testing QP-status + label-leakage fixes")
    print("=" * 60)
    test_qp_infeasible_detected()
    test_qp_feasible_ok()
    test_label_leakage_truncation()
    test_label_leakage_short()
    print("=" * 60)
    print("ALL TESTS PASSED")
    sys.exit(0)
