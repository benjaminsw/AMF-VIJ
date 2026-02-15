# Incremental sEMA Additions to AMF-VI v2.0.0
**Base:** `threeflows_amf_vi_weights_log.py` v2.0.0  
**Goal:** Add full sEMA Stage-2 weight update mechanism one step at a time to verify improvement at each increment.  
**Paper defaults:** `(τ, α, β, ε, M, B) = (1.1, 0.9, 1e-5, 1e-5, 2, 2000)`

---

## Current State: v2.0.0 (Baseline)

**Stage-2 weight update (current):**
```python
flow_log_probs = [flow.log_prob(val_batch).mean()]          # plain avg log-prob
normalized = softmax(flow_log_probs_tensor)                  # no temperature, no prior
π ← α * π + (1-α) * normalized                              # EMA only
# missing: log πk prior, floor, renorm, smoothing, multi-batch
```
**Missing vs paper:** τ, log πk prior, ε floor, β smoothing, M>1 batches

---

## Step 1 — Add Temperature τ → v2.1.0

**What changes:**
```python
# Before
normalized = softmax(flow_log_probs_tensor)

# After
tau = 1.1
normalized = softmax(flow_log_probs_tensor / tau)
```

**Why first:** Largest single impact — controls sharpness of responsibility allocation.  
Higher τ → flatter weights → more expert diversity → higher Neff.  
**Expected:** Smoother weight trajectories, reduced collapse risk on `rings`.

**Abbr:** `SEMA-TAU` | **Version:** `v2.1.0`

---

## Step 2 — Add log πk Prior Term → v2.2.0

**What changes:**
```python
# Before (v2.1.0)
normalized = softmax(flow_log_probs_tensor / tau)

# After
log_pi = torch.log(self.weights.clamp(min=1e-8))
normalized = softmax((flow_log_probs_tensor + log_pi) / tau)
```

**Why second:** Makes it proper posterior-weighted (EM-style) — incorporates current weight belief into responsibility estimate. Aligns with paper Eq. (22).  
**Expected:** More stable convergence; weights self-reinforce correctly-specialised experts.

**Abbr:** `SEMA-PRIOR` | **Version:** `v2.2.0`

---

## Step 3 — Add Floor + Renorm ε → v2.3.0

**What changes:**
```python
# After EMA update, add:
eps = 1e-5
self.weights.data = torch.clamp(self.weights.data, min=eps)
self.weights.data = self.weights.data / self.weights.data.sum()
```

**Why third:** Safety fix — prevents any expert being driven to exactly zero (collapse prevention). Critical for `BLR`/`BPR` where one expert dominates.  
**Expected:** No expert zeroed out; Neff stays > 1 throughout training.

**Abbr:** `SEMA-FLOOR` | **Version:** `v2.3.0`

---

## Step 4 — Add Uniform Smoothing β → v2.4.0

**What changes:**
```python
# After averaging r̄, before EMA:
beta = 1e-5
K = len(self.flows)
r_bar = (1 - beta) * r_bar + beta * (torch.ones(K) / K)
```

**Why fourth:** Regulariser — damps early transients and prevents premature weight collapse during first few epochs. Only matters if instability observed in prior steps.  
**Expected:** Smoother early dynamics; less oscillation on `rings`.

**Abbr:** `SEMA-SMOOTH` | **Version:** `v2.4.0`

---

## Step 5 — Add Multi-Batch Averaging M > 1 → v2.5.0

**What changes:**
```python
# Before: single val_batch per epoch
# After: average r̄ over M=2 batches
M = 2
r_bar = torch.zeros(K)
for _ in range(M):
    indices = torch.randperm(len(val_data))[:batch_size]
    val_batch = val_data[indices]
    # compute responsibilities per batch
    r_bar += responsibilities / M
# then apply smoothing + EMA as normal
```

**Why fifth:** Variance reduction — reduces Monte Carlo noise in responsibility estimates. Diminishing returns in 2D; more impactful for real Bayesian targets (BLR, BPR, Weibull).  
**Expected:** Reduced per-epoch weight oscillation on noisy targets like `rings`.

**Abbr:** `SEMA-MBATCH` | **Version:** `v2.5.0`

---

## Summary Table

| Step | Abbr | Version | Change | Primary Benefit |
|------|------|---------|--------|-----------------|
| 1 | `SEMA-TAU` | v2.1.0 | Add `τ=1.1` to softmax | Controls responsibility sharpness |
| 2 | `SEMA-PRIOR` | v2.2.0 | Add `log πk` prior term | EM-style posterior weighting |
| 3 | `SEMA-FLOOR` | v2.3.0 | Add `ε` floor + renorm | Collapse prevention |
| 4 | `SEMA-SMOOTH` | v2.4.0 | Add `β` uniform smoothing | Early transient damping |
| 5 | `SEMA-MBATCH` | v2.5.0 | Average over `M=2` batches | Responsibility variance reduction |

---

## Evaluation Checklist (after each step)

Track the following metrics on held-out test set to confirm improvement:

- [ ] **NLL** — primary metric (lower is better)
- [ ] **Neff** = exp(H(π)) — should stay > 1.5 on synthetics
- [ ] **Weight trajectory** — should be smooth, not oscillating
- [ ] **Final weights** — no expert driven to ~0 (collapse)
- [ ] **Stage-2 loss** — should decrease or stabilise

---

## Notes

- Test on at minimum: `multimodal`, `rings`, `bimodal` (diverse geometry coverage)
- Paper target Neff values: X-shaped 2.97, Multimodal 2.88, Rings 2.90, BLR 1.89
- If Step 4 (β smoothing) shows no change, it can be skipped — β=1e-5 is very small
- If Step 5 shows no change on 2D synthetics, reserve for real targets (BLR, BPR, Weibull)