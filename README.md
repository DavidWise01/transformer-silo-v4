# transformer-silo v4 — two silos in parallel, tested at the end

v3 trained one silo against a plain transformer and found the honest edges: the
silo wins the order-insensitive task (cheaper) and **collapses on the order task**
because it throws position away. v4 asks the obvious next thing — *"both silos ran
in parallel, tested at the end"* — and answers it with real training: **can two
parallel, complementary silos recover the order one silo lost?**

## The dual silo

Two silos run in parallel over the same context and only meet at the final head
("tested at the end"):

- **content silo** — k-means on the token embeddings → K intents. *Orderless* (v3's silo).
- **position silo** — the sequence chopped into K ordered bins, each intent the
  bin's mean embedding + a per-bin positional code. *Keeps order.*

The **same shared encoder** processes both branches (so the dual has ~one silo's
capacity — only the head grows from `d` to `2d` in), their pooled outputs are
**concatenated**, and a linear head classifies. Attention cost: `2·K²` pairs —
still half of the plain model's `N²`.

## What happened (seed 0, 800 train / 400 test, held out)

| task | chance | plain | content-silo | **dual** | attention |
|------|--------|-------|--------------|----------|-----------|
| **plurality** (order-insensitive) | 25% | 95.3% | 99.5% | **98.8%** | dual 2K²=32 vs N²=64 |
| **first** (order-sensitive) | 25% | 85.0% | 42.5% | **61.8%** | dual 2K²=32 vs N²=64 |

**Read it straight.** On the set task the dual just matches the single silos (a
second view neither helps nor hurts). On the order task the single silo collapsed
to 42.5% — and the dual **recovers most of the gap, to 61.8%**, because its second
branch keeps order. But the recovery is **partial, not full**: chunking into K bins
is coarser than exact positions, so the dual lands *between* the single silo and
the plain model (85.0%) — while using **half** the plain model's attention. A
second, complementary view buys back most of the order a single silo throws away,
cheaply — **but not all of it.**

## The honest caveats

- **Partial recovery, and it says so.** 61.8% is well short of plain's 85% — the
  page shows all three bars side by side so the gap is visible, not hidden.
- The **plurality "win"** is still k-means' home turf (a cluster-shaped task); don't
  read it as "silos are better."
- Synthetic probes, tiny models, one seed — a clean controlled ablation, not a
  leaderboard. Reproducible with `python train.py`.

## Verify first

```bash
python selftest.py    # gradient-checks BOTH the single and the DUAL backprop, + the recovery finding
python train.py       # retrain all three arms on both tasks -> results.json
```

The **dual gradient check** is v4's honesty anchor: the two-parallel-silos,
shared-encoder, merge-at-the-end model really trains by gradient descent (analytic
grads match numerical < 1e-5).

## Files

| File | Role |
|------|------|
| `model.py` | shared encoder + single & **dual** heads; hand-written, gradient-checked backward |
| `tasks.py` | the world, the content silo + the **position silo**, the two probes |
| `train.py` | Adam training of all three arms → `results.json` |
| `selftest.py` | grad checks (single + dual) + determinism + the recovery finding |
| `results.json` | the trained results the page reports |
| `index.html` | the three-arm results page — bars, curves, the recovery, the verdict |

The tetralogy: [v1](https://davidwise01.github.io/transformer-silo/) build ·
[v2](https://davidwise01.github.io/transformer-silo-v2/) measure ·
[v3](https://davidwise01.github.io/transformer-silo-v3/) train one silo · v4 two silos in parallel.

---
David Lee Wise / ROOT0 / TriPod LLC · CC-BY-ND-4.0
