"""
moment_leaf_test.py

One-off test of the order-2 moment relaxation (moment_leaf_bound.py) on
the worst loose birds-foot leaf, [0,1,2,4,5,7,8,10,11] (the 96.8% one-swap
trap): disk (order-1) bound 0.010890, true optimum 0.008969 (best 5 of 40
Riemannian restarts agree), incumbent/champion 0.009263.

Steps:
  1. Order 1 must reproduce the disk bound (sign change of the margin).
  2. Order 2 at tau = 0.008969 (achieved): margin must be >= 0 (validity).
  3. Order 2 at tau = champion * (1 + 1e-4): margin < 0 PROVES the leaf
     can't beat the champion.
  4. If affordable, bisect for the order-2 bound itself.

Saves: results/moment_leaf_test_summary.txt
Run directly: python -u moment_leaf_test.py
"""

import sys, os, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interactive_optimizer import build_birdsfoot_system, BIRDSFOOT_START_THETAS_DEG
from swap_search import SetEvaluator
from log_product_polish import ReducedKModel
from moment_leaf_bound import MomentLeafRelaxation

LEAF = [0, 1, 2, 4, 5, 7, 8, 10, 11]
ACHIEVED = 0.008969
CHAMPION = 0.009263
OUT = os.path.join(os.path.dirname(__file__), 'results', 'moment_leaf_test_summary.txt')

lines = []


def log(msg):
    print(msg, flush=True)
    lines.append(msg)


def main():
    s = build_birdsfoot_system(np.array(BIRDSFOOT_START_THETAS_DEG))
    tr = s.total_dofs - 6
    ev = SetEvaluator(s, tr, BIRDSFOOT_START_THETAS_DEG, seed=31)
    _, b_disk, ref = ev.bound(LEAF)
    mdl = ReducedKModel(s, LEAF, ref[LEAF], tr)
    A = [a for kc, ks in zip(mdl.Kcr, mdl.Ksr) for a in (kc, ks)]

    log(f"Order-2 moment relaxation test, birds-foot leaf {LEAF}")
    log(f"disk (order-1) bound {b_disk:.6f}; achieved {ACHIEVED:.6f}; champion {CHAMPION:.6f}")

    r1 = MomentLeafRelaxation(mdl.K0r, A, order=1)
    lo_e = r1.margin(b_disk * 0.999)[0]
    hi_e = r1.margin(b_disk * 1.001)[0]
    log(f"1. order 1: margin at 0.999x disk bound {lo_e:+.2e}, at 1.001x {hi_e:+.2e} "
        f"-> {'reproduces disk bound' if lo_e >= 0 > hi_e else 'MISMATCH'}")

    t0 = time.perf_counter()
    r2 = MomentLeafRelaxation(mdl.K0r, A, order=2)
    log(f"order 2: {r2.N} moments, M {r2.nb_M}x{r2.nb_M}, L {r2.size_L}x{r2.size_L} "
        f"(built in {time.perf_counter() - t0:.0f}s)")

    e, st, sec = r2.margin(ACHIEVED * (1 - 1e-4))
    log(f"2. order 2 at tau {ACHIEVED * (1 - 1e-4):.6f} (just below achieved): eps {e:+.3e} "
        f"({st}, {sec:.0f}s) -> {'valid' if e >= -1e-7 else 'INVALID'}")

    thresh = CHAMPION * (1 + 1e-4)
    e, st, sec = r2.margin(thresh)
    proved = e < 0
    log(f"3. order 2 at tau {thresh:.6f} (champion + tol): eps {e:+.3e} ({st}, {sec:.0f}s) "
        f"-> {'PROVED: leaf cannot beat the champion' if proved else 'not proved'}")

    if sec < 900:
        log("4. bisecting for the order-2 bound:")
        b2 = r2.bound(ACHIEVED * (1 - 1e-4), b_disk, rel_tol=2e-3, log=log)
        log(f"   order-2 bound {b2:.6f} ({100 * (b2 / ACHIEVED - 1):.1f}% above achieved; "
            f"disk bound was {100 * (b_disk / ACHIEVED - 1):.1f}%)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
