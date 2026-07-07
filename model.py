#!/usr/bin/env python3
"""v4 model: a shared transformer ENCODER (one attention block + MLP + weighted
mean-pool -> a pooled vector), with two heads:

  * single -- pooled -> linear -> C classes   (plain / one-silo arms)
  * dual   -- run the SAME encoder over TWO parallel views, concatenate the two
              pooled vectors, then linear -> C classes. This is "both silos ran
              in parallel, tested at the end": the branches only meet at the head.

The encoder weights are SHARED across the two dual branches, so the dual arm has
essentially the same capacity as one silo (only the head grows from d to 2d in) --
this isolates "does a second, complementary view help?" from "more parameters."

All backprop is hand-written and gradient-checked in selftest.py -- the honesty
anchor: the training is real.
"""
from __future__ import annotations
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def init_params(d, h, C, head_in=None, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.2
    def w(*shape): return rng.standard_normal(shape) * s
    hin = head_in if head_in is not None else d
    return {
        "Wq": w(d, d), "Wk": w(d, d), "Wv": w(d, d), "Wo": w(d, d),
        "W1": w(d, h), "b1": np.zeros(h), "W2": w(h, d), "b2": np.zeros(d),
        "Wc": w(hin, C), "bc": np.zeros(C),
    }


# ---------- the shared encoder: (X, w) -> pooled vector ----------
def encode(P, X, w=None):
    n, d = X.shape
    if w is None:
        w = np.ones(n)
    wn = w / w.sum()
    Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]
    S = (Q @ K.T) / np.sqrt(d)
    A = softmax(S, axis=1)
    Ctx = A @ V
    Attn = Ctx @ P["Wo"]
    Z1 = X + Attn
    Hpre = Z1 @ P["W1"] + P["b1"]
    H = np.maximum(0.0, Hpre)
    M = H @ P["W2"] + P["b2"]
    Z2 = Z1 + M
    p = (Z2 * wn[:, None]).sum(axis=0)
    cache = dict(X=X, K=K, V=V, Q=Q, A=A, Ctx=Ctx, Z1=Z1, Hpre=Hpre, H=H, wn=wn, d=d)
    return p, cache


def encode_backward(P, cache, dp):
    """Grads for the shared encoder params, given d(loss)/d(pooled)."""
    g = {}
    dZ2 = np.outer(cache["wn"], dp)
    dZ1 = dZ2.copy()
    dM = dZ2
    g["W2"] = cache["H"].T @ dM
    g["b2"] = dM.sum(axis=0)
    dH = dM @ P["W2"].T
    dHpre = dH * (cache["Hpre"] > 0)
    g["W1"] = cache["Z1"].T @ dHpre
    g["b1"] = dHpre.sum(axis=0)
    dZ1 += dHpre @ P["W1"].T
    dAttn = dZ1
    g["Wo"] = cache["Ctx"].T @ dAttn
    dCtx = dAttn @ P["Wo"].T
    dA = dCtx @ cache["V"].T
    dV = cache["A"].T @ dCtx
    dS = cache["A"] * (dA - (dA * cache["A"]).sum(axis=1, keepdims=True))
    dS /= np.sqrt(cache["d"])
    dQ = dS @ cache["K"]
    dK = dS.T @ cache["Q"]
    g["Wq"] = cache["X"].T @ dQ
    g["Wk"] = cache["X"].T @ dK
    g["Wv"] = cache["X"].T @ dV
    return g


def _zeros_like(P):
    return {k: np.zeros_like(v) for k, v in P.items()}


# ---------- single-view arm (plain / one silo) ----------
def loss_and_grad_single(P, X, y, w=None):
    p, cache = encode(P, X, w)
    logits = p @ P["Wc"] + P["bc"]
    probs = softmax(logits)
    loss = -np.log(probs[y] + 1e-12)
    dlogits = probs.copy(); dlogits[y] -= 1.0
    g = _zeros_like(P)
    g["Wc"] = np.outer(p, dlogits)
    g["bc"] = dlogits
    dp = P["Wc"] @ dlogits
    ge = encode_backward(P, cache, dp)
    for k in ge:
        g[k] += ge[k]
    return loss, g


def predict_single(P, X, w=None):
    p, _ = encode(P, X, w)
    return int(np.argmax(p @ P["Wc"] + P["bc"]))


# ---------- dual-view arm: two parallel silos, merged at the end ----------
def loss_and_grad_dual(P, XA, wA, XB, wB, y):
    pA, cA = encode(P, XA, wA)
    pB, cB = encode(P, XB, wB)
    cat = np.concatenate([pA, pB])                      # (2d,)
    logits = cat @ P["Wc"] + P["bc"]
    probs = softmax(logits)
    loss = -np.log(probs[y] + 1e-12)
    dlogits = probs.copy(); dlogits[y] -= 1.0
    g = _zeros_like(P)
    g["Wc"] = np.outer(cat, dlogits)
    g["bc"] = dlogits
    dcat = P["Wc"] @ dlogits                            # (2d,)
    d = pA.shape[0]
    dpA, dpB = dcat[:d], dcat[d:]
    for ge in (encode_backward(P, cA, dpA), encode_backward(P, cB, dpB)):
        for k in ge:
            g[k] += ge[k]                               # shared encoder grads accumulate
    return loss, g


def predict_dual(P, XA, wA, XB, wB):
    pA, _ = encode(P, XA, wA)
    pB, _ = encode(P, XB, wB)
    cat = np.concatenate([pA, pB])
    return int(np.argmax(cat @ P["Wc"] + P["bc"]))
