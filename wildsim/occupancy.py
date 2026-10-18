"""Single-season occupancy estimators applied to simulated detection histories.

* naive        -- proportion of sites with at least one record
* mackenzie    -- MacKenzie et al. (2002): imperfect detection, no false positives
* royle_link   -- Royle & Link (2006): imperfect detection AND false positives,
                  identified by the constraint p11 > p10

Comparing the three against the known true occupancy shows how much of a
detector's error a standard ecological analysis absorbs, and how much leaks
into biased ecological conclusions.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


def to_occasions(y_daily, occasion_days):
    """Collapse (S, D) daily records into (S, J) occasions of `occasion_days` each."""
    S, D = y_daily.shape
    J = D // occasion_days
    y = y_daily[:, : J * occasion_days].reshape(S, J, occasion_days)
    return y.max(axis=2)


def naive(y):
    return float((y.sum(axis=1) > 0).mean())


def _nll_mackenzie(theta, d, J, any_det):
    lpsi, lp = theta
    log_psi, log_1mpsi = log_expit(lpsi), log_expit(-lpsi)
    log_p, log_1mp = log_expit(lp), log_expit(-lp)
    l_occ = log_psi + d * log_p + (J - d) * log_1mp
    l_all0 = np.logaddexp(l_occ, log_1mpsi)
    return -np.where(any_det, l_occ, l_all0).sum()


def mackenzie(y):
    S, J = y.shape
    d = y.sum(axis=1)
    if d.sum() == 0:
        return {"psi": 0.0, "p": np.nan}
    best = None
    for start in ([0.0, 0.0], [2.0, -1.0], [-1.0, 1.0]):
        r = minimize(_nll_mackenzie, start, args=(d, J, d > 0), method="L-BFGS-B",
                     bounds=[(-10, 10), (-10, 10)])
        if best is None or r.fun < best.fun:
            best = r
    return {"psi": float(expit(best.x[0])), "p": float(expit(best.x[1]))}


def _nll_royle_link(theta, d, J):
    lpsi, a, b = theta
    p10 = expit(a)
    p11 = p10 + (1 - p10) * expit(b)          # enforces p11 > p10
    p11 = np.clip(p11, 1e-9, 1 - 1e-9)
    l1 = log_expit(lpsi) + d * np.log(p11) + (J - d) * np.log1p(-p11)
    l0 = log_expit(-lpsi) + d * np.log(p10) + (J - d) * np.log1p(-p10)
    return -np.logaddexp(l1, l0).sum()


def royle_link(y):
    S, J = y.shape
    d = y.sum(axis=1)
    if d.sum() == 0:
        return {"psi": 0.0, "p11": np.nan, "p10": np.nan}
    best = None
    for start in ([0.0, -3.0, 1.0], [1.0, -4.0, 2.0], [-1.0, -2.5, 0.5]):
        r = minimize(_nll_royle_link, start, args=(d, J), method="L-BFGS-B",
                     bounds=[(-10, 10), (-10, 3), (-10, 10)])
        if best is None or r.fun < best.fun:
            best = r
    lpsi, a, b = best.x
    p10 = expit(a)
    return {"psi": float(expit(lpsi)), "p10": float(p10),
            "p11": float(p10 + (1 - p10) * expit(b))}
