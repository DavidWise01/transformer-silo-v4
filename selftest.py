#!/usr/bin/env python3
"""Verify-first self-test for v4. Proves the training is REAL and the headline
finding is what actually happens, with no network:
(1) GRADIENT CHECK -- both the single-view and the DUAL (two parallel silos,
    shared encoder, merged head) backprop match numerical gradients < 1e-5;
(2) determinism;
(3) it LEARNS -- all three arms beat chance on the order-insensitive task;
(4) the FINDING -- on the order-SENSITIVE task the DUAL recovers well above the
    single content silo (its position branch keeps order). The full, authoritative
    'partial recovery' ordering (silo < dual < plain) is the deterministic result
    in results.json, reproducible with `python train.py`.
"""
from __future__ import annotations
import numpy as np
from model import (init_params, loss_and_grad_single, loss_and_grad_dual, encode, softmax)
from tasks import make_world, dataset, D, G
from train import train_arm

fails = 0
def check(cond, msg):
    global fails
    print(("ok  · " if cond else "FAIL· ") + msg)
    fails += 0 if cond else 1


def _gradcheck(P, floss, g):
    eps = 1e-6; mr = 0.0
    for nm in P:
        W = P[nm]; num = np.zeros_like(W); it = np.nditer(W, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index; o = W[i]
            W[i] = o + eps; lp = floss(); W[i] = o - eps; lm = floss(); W[i] = o
            num[i] = (lp - lm) / (2 * eps); it.iternext()
        rel = np.abs(num - g[nm]).max() / (np.abs(num).max() + np.abs(g[nm]).max() + 1e-12)
        mr = max(mr, rel)
    return mr

rng = np.random.default_rng(5); d, h, C = 6, 10, 4
# 1a. single-view gradient check
Ps = init_params(d, h, C, head_in=d, seed=1)
Xs = rng.standard_normal((5, d)); ws = rng.random(5) + 0.3; ys = 2
_, gs = loss_and_grad_single(Ps, Xs, ys, ws)
def fs():
    p, _ = encode(Ps, Xs, ws); lg = p @ Ps["Wc"] + Ps["bc"]; return -np.log(softmax(lg)[ys] + 1e-12)
check(_gradcheck(Ps, fs, gs) < 1e-5, "single-view gradient check passes (training is real)")

# 1b. DUAL gradient check (shared encoder over two views, merged head)
Pd = init_params(d, h, C, head_in=2 * d, seed=2)
XA = rng.standard_normal((5, d)); wA = rng.random(5) + 0.3
XB = rng.standard_normal((4, d)); wB = rng.random(4) + 0.3
_, gd = loss_and_grad_dual(Pd, XA, wA, XB, wB, ys)
def fd():
    pA, _ = encode(Pd, XA, wA); pB, _ = encode(Pd, XB, wB)
    lg = np.concatenate([pA, pB]) @ Pd["Wc"] + Pd["bc"]; return -np.log(softmax(lg)[ys] + 1e-12)
check(_gradcheck(Pd, fd, gd) < 1e-5, "DUAL gradient check passes (two parallel silos, merged -- real)")

world = make_world(seed=0)
def quick(task, arm, seed=0, epochs=12):
    data = dataset(world, task, n_train=300, n_test=200, seed=100)
    return train_arm(world, data, arm, seed=seed, record=False, epochs=epochs)[1]

# 2. Determinism.
check(quick("plurality", "dual") == quick("plurality", "dual"), "training is deterministic")

# 3. It LEARNS: all three arms beat chance on the order-insensitive task.
ch = 1.0 / G
for arm in ("plain", "silo", "dual"):
    a = quick("plurality", arm)
    check(a > ch + 0.2, f"{arm} learns plurality well above chance ({a:.2f})")

# 4. THE FINDING: on the ORDER task the DUAL recovers well above the single silo.
fs_silo = quick("first", "silo", epochs=30)
fs_dual = quick("first", "dual", epochs=30)
check(fs_dual - fs_silo > 0.1, f"DUAL recovers order over the single content silo (+{fs_dual-fs_silo:.2f}: {fs_silo:.2f}->{fs_dual:.2f})")
check(fs_dual > ch + 0.1, f"the DUAL is clearly above chance on the order task ({fs_dual:.2f} vs {ch:.2f})")

print("\n" + ("SOME CHECKS FAILED" if fails else "all transformer-silo-v4 checks passed"))
raise SystemExit(1 if fails else 0)
