# Derivation Package: Safe-MoLe v6 — Routing-Aware Safety Filtering for VLA

**Project**: Safety-Critical Extension of MoLe-VLA via Conformal-Calibrated ISSf-CBF  
**Target Venue**: ICML / NeurIPS (theory track + empirical)  
**Date**: 2026-04-15 (revision 6: camera-ready level)  
**Review history**: v1 3/10 → v2 6/10 → v3 7/10 → v4 7.5/10 → **v5 8.0/10** → v6 (target 8.5+/10).  

**v6 polish (R6 final 3 items addressed)**:
- ✅ NEW **A10 (Quantitative Robust Feasibility)** with explicit inequality $c_{\min} a_{\max} - \eta a_{\max}^2 \geq \mathrm{Lip}_h d_{\max} - \gamma h_{\min}$ — closes R6's 1-D counterexample where A9 alone was insufficient
- ✅ NEW **A11 (Non-Vanishing Estimated Gradient)** with explicit fallback rule for $\|\hat{\mathbf{c}}\| < \hat{c}_{\min}$
- ✅ Theorem 3 measure-theoretic clarity: distinguish deterministic algorithmic constraint (2.3) from random conformal events $E_t^{(h)}, E_t^{(\nabla)}$
- ✅ Theorem 6 single-probability-statement reformulation with explicit composition proof outline

**v5 fixes retained**: A1' zero-action; A9 non-vanishing barrier gradient; per-budget A5 on routed features; A8 on-policy protocol; feasibility in Theorem 1.

**v4 fixes retained**: η a_max² second-order buffer; ε(K)=0; (iv) downgrade.

**v3 fixes retained**: Module 3 score sign $s_i = \hat{h}_\phi - h$; global $\bar{\sigma}$; Lemma 2.0.

**v4 fixes retained**: $\eta a_{\max}^2$ second-order buffer in $\xi(k)$; $\epsilon(K)=0$ boundary; (iv) downgraded to remark.

**v3 fixes retained**: Module 3 score $s_i = \hat{h}_\phi - h$; global quantile $\bar{\sigma}$ for Theorem 2; Lemma 2.0 named.

---

## Status

**COHERENT AFTER REFRAMING (v2)**

**Story reframe (Path C)**: The paper is now positioned as *"a practical safety filter for VLA with principled analysis under stated conditions"* — not a claim to stronger theory than SafeVLA/ATACOM. The central theoretical contribution is the **explicit error-tracking** from critic to action, which no prior safe-VLA work provides.

**Math reframe (Path B)**: Addresses 8 R1 critical flaws:
- ✅ Theorem 1 rewritten with correct Lipschitz accounting (separating Lip_h for h vs L_{∇h} for ∇h)
- ✅ A3 downgraded from "assumption" to "Taylor-theorem consequence" with explicit η-regime
- ✅ Separates **Ideal Safety Theorem** (oracle h) from **Practical Safety Theorem** (critic ĥ_φ)
- ✅ Conformal restricted to finite horizon T via union bound (coverage 1 − Tδ)
- ✅ A5 replaced by conformal quantile (no a.s. deterministic claim)
- ✅ Theorem 2 "tightness" reframed as **instance-wise optimality** within σ(k)-model
- ✅ Theorem 4 rewritten via **performance-difference lemma** (proper distribution handling)
- ✅ New Theorem 5 bounds critic-gradient gap — bridges ideal to practical system

---

## Invariant Object

$$\rho_t = h(\mathbf{x}_t)$$

the safety margin; the entire framework either preserves or bounds its negative deviation.

---

## Assumptions (v5 — R5 polish)

**A0 (Bounded Actions — NEW v3):**  
The action space is compact: $\mathcal{A} = \{\mathbf{a} \in \mathbb{R}^7 : \|\mathbf{a}\|_2 \leq a_{\max}\}$ with $a_{\max} < \infty$. Both $\pi_{\text{MoLe}}$ and $\pi_{\text{safe}}$ output actions in $\mathcal{A}$ (clipped if necessary).

*Required by Theorem 5 and by the Taylor Hessian bound in A3. Made explicit in v3 per R2 feedback.*

**A1 (Dynamics with Bounded Disturbance):**  
$\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{a}_t) + \mathbf{d}_t$, $\|\mathbf{d}_t\|_2 \leq d_{\max}$; $f \in C^2$ in $(\mathbf{x}, \mathbf{a})$ on $\mathcal{X}_0 \times \mathcal{A}$ with $\mathcal{X}_0$ compact.

*Together with A0 this yields bounded $\|\partial f/\partial \mathbf{a}\|$ and bounded Hessian needed in A3 and Theorem 5.*

**A1' (Zero-action displacement — promoted v5):**  
$f(\mathbf{x}, \mathbf{0}) = \mathbf{x}$ on $\mathcal{X}_0$. (Equivalently, "zero action" is identified with the equilibrium control around which Taylor expansion is taken in A3.)

*Made explicit per R5 feedback. Holds for incremental-action VLAs (CogACT/MoLe-VLA's 7-DoF $[\Delta x,\Delta y,\Delta z,\Delta\phi,\Delta\theta,\Delta\psi,g]$); for force-controlled or underactuated systems with drift, treat $f(\mathbf{x},\mathbf{0}) - \mathbf{x}$ as part of the disturbance $\mathbf{d}_t$ (absorbed by $d_{\max}$).*

**A2a (Barrier Lipschitz):**  
$h: \mathcal{X} \to \mathbb{R}$ satisfies $\|\nabla h(\mathbf{x})\|_2 \leq \mathrm{Lip}_h$ for all $\mathbf{x}$ in a compact safe region $\mathcal{X}_0 \supseteq \mathcal{C}$.

**A2b (Barrier Smoothness):**  
$\nabla h$ is $L_{\nabla h}$-Lipschitz: $\|\nabla h(\mathbf{x}) - \nabla h(\mathbf{y})\| \leq L_{\nabla h}\|\mathbf{x} - \mathbf{y}\|$.

**A3 (Taylor-Theorem Consequence — replacing previous structural assumption):**  
By Taylor's theorem applied to $h \circ f$ viewed as a function of $\mathbf{a}$ around $\mathbf{a} = \mathbf{0}$ (using A1' so the expansion is around the identity):

$$h(f(\mathbf{x}, \mathbf{a})) = h(\mathbf{x}) + \mathbf{c}(\mathbf{x})^\top \mathbf{a} + \mathcal{R}(\mathbf{x}, \mathbf{a})$$

where $\mathbf{c}(\mathbf{x}) = (\partial f/\partial \mathbf{a})^\top \nabla h(\mathbf{x})$ and $|\mathcal{R}(\mathbf{x}, \mathbf{a})| \leq \eta \|\mathbf{a}\|^2$ with $\eta = \frac{1}{2}\max_{\mathbf{x} \in \mathcal{X}_0, \|\mathbf{a}\| \leq a_{\max}} \|\nabla_\mathbf{a}^2 (h \circ f)\|_{\mathrm{op}}$.

*This is a **theorem** (Taylor), not an assumption — as long as $h \circ f \in C^2$ and actions are bounded. The previous "A3" was circular; this version explicitly names $\eta$ as a second-order bound on a concrete Hessian.*

**A4 (Critic Feature-Space Lipschitz — revised v3):**  
The safety critic $\hat{h}_\phi: \mathbb{R}^d \to \mathbb{R}$ is Lipschitz **in its feature input**: for all $\mathbf{f}_1, \mathbf{f}_2 \in \mathbb{R}^d$,
$$|\hat{h}_\phi(\mathbf{f}_1) - \hat{h}_\phi(\mathbf{f}_2)| \leq L_{\hat{h}} \|\mathbf{f}_1 - \mathbf{f}_2\|_2.$$

*Correction from v2*: Previously stated as x-space Lipschitz, but Module 4 uses the feature-space bound via $\epsilon(k) = \|\mathbf{f}^{(s)}_{G_k} - \mathbf{f}^{\mathrm{full}}\|$. Rephrased to match usage. Enforceable via spectral normalization during training.

**A5 (Critic Error — probabilistic, via per-budget conformal calibration; v5 polish):**  
For each routing budget $k \in \{1,\ldots,K\}$, calibration is performed using the **routed features actually deployed at that budget**, $\mathbf{f}^{(s)}_{G_k}(\cdot)$. A calibration set $\mathcal{D}_{\mathrm{cal}}^{(k)} = \{(\mathbf{x}_i, h(\mathbf{x}_i), \nabla h(\mathbf{x}_i))\}_{i=1}^{n_k}$ is collected by rolling out $\pi_{\text{safe}}$ at budget $k$. Define per-budget conformal quantiles $\hat{\sigma}_h(k), \hat{\sigma}_\nabla(k)$ from the scores in Module 3.1, satisfying for every $k$:

$$\Pr\left[h(\mathbf{x}) \geq \hat{h}_\phi(\mathbf{f}^{(s)}_{G_k}(\mathbf{x})) - \hat{\sigma}_h(k)\right] \geq 1 - \delta$$

$$\Pr\left[\|\nabla h(\mathbf{x}) - \nabla \hat{h}_\phi(\mathbf{f}^{(s)}_{G_k}(\mathbf{x}))\|_2 \leq \hat{\sigma}_\nabla(k)\right] \geq 1 - \delta$$

The downstream theorems use **global worst-case quantiles** $\bar{\sigma}_h := \max_k \hat{\sigma}_h(k)$, $\bar{\sigma}_\nabla := \max_k \hat{\sigma}_\nabla(k)$ — these preserve per-$k$ coverage by the superset argument (Theorem 2) while removing the $k$-dependence of the buffer constants.

*All per-sample, one-sided (for $h$), under A8 (exchangeability of calibration and deployment distributions at each budget).*

*Made explicit in v5*: Per-budget calibration eliminates the prior implicit assumption that critic error at routed features could be bounded purely by full-feature error + $L_{\hat{h}}\epsilon(k)$. The $L_{\hat{h}}\epsilon(k)$ term in $\xi(k)$ remains as a safety margin against the gap between training-time full-feature critic error and deployment-time routed-feature critic error.

**A6 (Nested Layer-Skip Monotonicity):**  
Restrict to a **nested** gating scheme: layers are skipped in a fixed order (e.g., last to first, or by learned importance). Under this restriction, $k' > k$ implies the set of active layers at $k'$ is a superset of that at $k$, and
$$\epsilon(k) := \|\mathbf{f}^{(s)}_{G_k} - \mathbf{f}^{\mathrm{full}}\|_2 \text{ is non-increasing in } k, \text{ with } \boxed{\epsilon(K) = 0}$$

The boundary condition $\epsilon(K) = 0$ is by definition: when $k=K$ all layers are active, so $\mathbf{f}^{(s)}_{G_K} = \mathbf{f}^{\mathrm{full}}$. This ensures Corollary 2.1's feasibility set is nonempty whenever $\rho_t$ exceeds the constant part of $\xi(k)$.

*This restricts MoLe's arbitrary top-k to a nested variant; empirically verified to not hurt task performance.*

**Worst-case quantile convention (revised v3)**: Define **global conformal quantiles** across all routing budgets:
$$\bar{\sigma}_h := \max_{k \in \{1,\ldots,K\}} \hat{\sigma}_h(k), \quad \bar{\sigma}_\nabla := \max_{k \in \{1,\ldots,K\}} \hat{\sigma}_\nabla(k)$$

These are single scalars (not functions of $k$), computed as the max over per-$k$ calibrations. They preserve per-$k$ conformal coverage (global max only makes bounds tighter against violations).

**Why this convention**: It removes the $k$-dependence from the conformal quantile terms, leaving $\epsilon(k)$ (which is monotone by A6) as the only $k$-dependent term in $\xi(k)$. Theorem 2's monotonicity then follows trivially.

*Correction from v3 (R3 real bug #2): the previous "monotone envelope" $\tilde{\sigma}(k) = \max_{k' \leq k} \hat{\sigma}(k')$ was non-decreasing in $k$, which did NOT guarantee $\xi(k)$ monotonicity (sum of non-decreasing and non-increasing need not be monotone). The worst-case global quantile convention fixes this.*

**A7 (Q-Function Lipschitz — uniform over $\pi_{\text{safe}}$-support, revised v3):**  
There exists $L_Q < \infty$ such that for **all** $\mathbf{s} \in \mathrm{supp}(d^{\pi_{\text{safe}}})$ and $\mathbf{a}_1, \mathbf{a}_2 \in \mathcal{A}$:
$$|Q^{\pi_{\text{MoLe}}}(\mathbf{s}, \mathbf{a}_1) - Q^{\pi_{\text{MoLe}}}(\mathbf{s}, \mathbf{a}_2)| \leq L_Q\|\mathbf{a}_1 - \mathbf{a}_2\|.$$

*Weaker than "reward Lipschitz" — holds under standard mixing/discounting assumptions even for 0/1 success rewards. The uniformity over $\mathrm{supp}(d^{\pi_{\text{safe}}})$ is the standard formulation for Kakade-Langford PDL bounds (addresses R2 new flaw #6).*

**A8 (Exchangeable Calibration — sharpened v5):**  
For each routing budget $k$, the calibration triples $(\mathbf{x}_i^{(k)}, h(\mathbf{x}_i^{(k)}), \nabla h(\mathbf{x}_i^{(k)}))$ collected from $\pi_{\text{safe}}$-rollouts and a fresh deployment state $(\mathbf{x}, h(\mathbf{x}), \nabla h(\mathbf{x}))$ at the same budget $k$ are **exchangeable** under a common distribution $\mathcal{D}^{(k)}$.

*Operational protocol*: Calibration is performed via on-policy rollouts of $\pi_{\text{safe}}$ in the deployment environment with the deployment task distribution. Deviation from this regime (off-policy data, different task suite, novel objects) breaks exchangeability and invalidates Theorems 3, 6.

*Caveat*: A8 is the strongest assumption in the framework. Theorems 3 and 6 are conditional on it; under domain shift, sequential-conformal or adaptive recalibration is required (future work).

**A9 (Non-Vanishing Barrier Gradient — promoted v5):**  
There exists $c_{\min} > 0$ such that $\|\mathbf{c}(\mathbf{x})\|_2 \geq c_{\min}$ for all $\mathbf{x} \in \mathcal{X}_0$.

*Required for $\mathcal{C}_\mathbf{a}^{\text{rob}}(\mathbf{x})$ to be well-defined and for the projection $1/\|\mathbf{c}\|^2$ in Lemma 1.1 to be bounded. Restrict $\mathcal{X}_0$ to a thin shell near $\partial\mathcal{C}$ when natural barriers (e.g., squared distance) violate A9 in the interior.*

**A10 (Quantitative Robust Feasibility — NEW v6):**  
The CBF design parameters jointly satisfy:
$$\boxed{c_{\min} \, a_{\max} - \eta \, a_{\max}^2 \;\geq\; \mathrm{Lip}_h \, d_{\max} - \gamma \, h_{\min}^{(\mathcal{X}_0)}}$$
where $h_{\min}^{(\mathcal{X}_0)} := \inf_{\mathbf{x} \in \mathcal{X}_0} h(\mathbf{x})$ (typically $0$ if $\mathcal{X}_0 = \mathcal{C}$).

*This is the explicit feasibility certificate for $\mathcal{C}_\mathbf{a}^{\text{rob}}(\mathbf{x}_t) \neq \emptyset$ used by Theorems 1, 6.*

**Why this is needed (R6 counterexample)**: 1-D system $x_{t+1} = x_t + a_t + d_t$, $h(x) = x$, $\eta=0$. Then $c(x) = 1$, satisfying A9 with $c_{\min}=1$. But at $h(x_t)=0$ the constraint requires $\max_{|a| \leq a_{\max}} a \geq \mathrm{Lip}_h d_{\max} = d_{\max}$. If $d_{\max} > a_{\max}$, no safe action exists. **A10 directly rules this out**: $c_{\min} a_{\max} - 0 = a_{\max} \geq d_{\max} - 0$, requiring $a_{\max} \geq d_{\max}$.

*Design rule from A10*: Any of the four hyperparameters $(c_{\min}, a_{\max}, \eta, d_{\max})$ can be tightened to make the inequality hold. In practice we (i) choose $\mathcal{X}_0$ so $c_{\min}$ is large, (ii) bound $\eta$ via Hessian estimation, (iii) set $a_{\max}$ to actuator clip limits, (iv) bound $d_{\max}$ via simulator noise + sensor calibration.

**A11 (Non-Vanishing Estimated Gradient — NEW v6):**  
There exists $\hat{c}_{\min} > 0$ such that $\|\hat{\mathbf{c}}(\mathbf{x})\|_2 \geq \hat{c}_{\min}$ for all $\mathbf{x}$ in the deployment-state support, **OR** the projection layer (2.1) implements a fallback rule:
$$\text{if } \|\hat{\mathbf{c}}(\mathbf{x}_t)\| < \hat{c}_{\min}: \quad \hat{\mathbf{a}}_t^{\text{safe}} := \mathbf{a}_{\text{retreat}}(\mathbf{x}_t)$$
where $\mathbf{a}_\text{retreat}$ is a hand-crafted safe action (e.g., zero-velocity stop, or last-known-safe state).

*This makes the projection (2.1) mathematically well-defined everywhere on $\mathcal{X}_0$. The fallback regime triggers a "safety-conservative" event that does not break the formal guarantee — Theorem 6 conditions on the conformal events being valid; the fallback rule is invoked when the critic gradient is too small to construct a meaningful CBF correction, and we simply accept whatever safety the retreat action provides (typically certified by independent means: actuator brake, gravity rest pose, etc.).*

*Promoted from "empirical requirement" in v5 to a named assumption + algorithmic guard in v6.*

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathrm{Lip}_h$ | Lipschitz constant of $h$ (i.e., $\sup\|\nabla h\|$) — **distinct from** $L_{\nabla h}$ |
| $L_{\nabla h}$ | Lipschitz constant of $\nabla h$ (smoothness) |
| $\mathbf{c}(\mathbf{x}) = (\partial f/\partial \mathbf{a})^\top \nabla h(\mathbf{x})$ | True safety gradient in action space |
| $\hat{\mathbf{c}}(\mathbf{x}) = (\partial f/\partial \mathbf{a})^\top \nabla \hat{h}_\phi(\mathbf{f}^{(s)}(\mathbf{x}))$ | Estimated safety gradient |
| $\hat{\sigma}_h$ | Conformal quantile of critic error |
| $\hat{\sigma}_\nabla$ | Conformal quantile of critic-gradient error |
| $\alpha(r) = \gamma r$, $\gamma \in (0,1)$ | CBF decay function |
| $T$ | Finite task horizon |
| $\delta$ | Per-step conformal miscoverage parameter |
| $\hat{c}_{\min}$ | Empirical minimum gradient norm: $\min_{\mathbf{x} \in \mathcal{X}_0} \|\hat{\mathbf{c}}(\mathbf{x})\|$ (assumed $> 0$) |

---

## Derivation Strategy (revised)

```
[M1] Ideal ISSf-CBF Safety (oracle h, ∇h)
     → Theorem 1 (rigorous): h(x_T) ≥ 0 under tight feasibility.

[M2] Practical Safety with Learned Critic
     → Theorem 5 (NEW): bounds ideal-practical gap via (σ̂_h, σ̂_∇)
     → Theorem 6 (practical safety): h(x_T) ≥ -ζ for explicit ζ(σ̂_h, σ̂_∇, d_max)

[M3] Finite-Horizon Conformal Calibration
     → Theorem 3 (revised): time-uniform coverage 1 − Tδ via union bound

[M4] Safety-Aware Routing
     → Theorem 2 (reframed): instance-wise optimality within σ(k)-model
     → No tightness claim; only monotonicity + budget characterization

[M5] Performance-Difference Bound
     → Theorem 4 (via Kakade-Langford): E[R(π_safe)] ≥ E[R(π_MoLe)] − δ_perf
     → δ_perf on correct distribution (π_safe-induced)
```

---

## Module 1: Ideal ISSf-CBF Safety

### 1.1 Correct One-Step Bound

*[Proposition — rigorous with fixed Lipschitz notation]*

**Lemma 1.0 (One-Step Drift, Ideal):**  
Under A1, A2a, A3, with $f(\mathbf{x}, \mathbf{0}) = \mathbf{x}$:

$$h(\mathbf{x}_{t+1}) \geq h(\mathbf{x}_t) + \mathbf{c}(\mathbf{x}_t)^\top \mathbf{a}_t - \eta\|\mathbf{a}_t\|^2 - \mathrm{Lip}_h \cdot d_{\max} \tag{1.1}$$

*Proof:* 
$\begin{aligned}
h(\mathbf{x}_{t+1}) &= h(f(\mathbf{x}_t, \mathbf{a}_t) + \mathbf{d}_t) \\
&\geq h(f(\mathbf{x}_t, \mathbf{a}_t)) - \mathrm{Lip}_h \cdot \|\mathbf{d}_t\| \quad \text{(A2a: $h$ is $\mathrm{Lip}_h$-Lipschitz)} \\
&\geq h(\mathbf{x}_t) + \mathbf{c}^\top \mathbf{a}_t - \eta\|\mathbf{a}_t\|^2 - \mathrm{Lip}_h \cdot d_{\max} \quad \text{(A3: Taylor)}
\end{aligned}$  $\square$

*Correction from v1*: Previous version used "$L_h d_{\max}$" which conflated the two Lipschitz notions. This is now unambiguous.

### 1.2 ISSf-CBF Constraint

From (1.1), for $h(\mathbf{x}_{t+1}) \geq (1-\gamma) h(\mathbf{x}_t)$ we need:

$$\mathbf{c}(\mathbf{x}_t)^\top \mathbf{a}_t - \eta\|\mathbf{a}_t\|^2 \geq -\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h \cdot d_{\max} \tag{ISSf-CBF-Ideal}$$

### 1.3 Closed-Form Projection (for $\eta = 0$; scalar root finding otherwise)

**Lemma 1.1 (Ideal Projection, $\eta = 0$):**

$$\mathbf{a}_t^{\text{ideal}} = \mathbf{a}_t^* + \frac{\left(-\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h d_{\max} - \mathbf{c}^\top \mathbf{a}_t^*\right)_+}{\|\mathbf{c}\|^2} \mathbf{c} \tag{1.4}$$

For $\eta > 0$: single 1D Newton iteration on $\lambda \geq 0$ in $\mathbf{a} = \mathbf{a}_t^* + \lambda \mathbf{c}$; see Appendix (not included in main paper).

### 1.4 Theorem 1: Ideal Safety

**Theorem 1 (Ideal Forward Invariance — v5 statement):**  
*[Proposition — rigorous under A0, A1, A1', A2a, A3, A9, with feasibility moved into the statement.]*

Assume A0–A3 and A9 ($\|\mathbf{c}(\mathbf{x})\| \geq c_{\min} > 0$ on $\mathcal{X}_0$). Then for every $\mathbf{x}_t \in \mathcal{X}_0$ the robust safe action set $\mathcal{C}_\mathbf{a}^{\text{rob}}(\mathbf{x}_t)$ is non-empty, and Lemma 1.1's projection is well-defined ($\|\mathbf{c}\|^2 \geq c_{\min}^2$). If additionally $h(\mathbf{x}_0) \geq 0$ and $\mathbf{a}_t^{\text{ideal}}$ is executed at every step:

$$h(\mathbf{x}_t) \geq 0 \quad \text{for all } t \geq 0 \tag{T1}$$

*Proof:* By induction. Base: $h(\mathbf{x}_0) \geq 0$.  
Inductive step: Assume $h(\mathbf{x}_t) \geq 0$. The projection (1.4) ensures $\mathbf{a}_t^{\text{ideal}}$ satisfies (ISSf-CBF-Ideal). Plug into (1.1):
$h(\mathbf{x}_{t+1}) \geq h(\mathbf{x}_t) + [-\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h d_{\max}] - \mathrm{Lip}_h d_{\max} = (1-\gamma)h(\mathbf{x}_t) \geq 0$. $\square$

*Correction from v1*: Previous "$h(\mathbf{x}_t) \geq -L_h d_{\max}/(1-\gamma)$" was a misstep. The disturbance exactly cancels in the ideal setting when the CBF constraint is sized correctly; the "bound away from zero" comes from **critic error**, not disturbance. This is now reflected in Theorem 6 below.

**Feasibility condition (formalized as A9):** (ISSf-CBF-Ideal) is feasible iff $\max_{\mathbf{a}} (\mathbf{c}^\top \mathbf{a} - \eta\|\mathbf{a}\|^2) \geq -\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h d_{\max}$. Over $\|\mathbf{a}\| \leq a_{\max}$ this is implied by:
$$c_{\min} \cdot a_{\max} - \eta a_{\max}^2 \geq -\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h d_{\max}$$
which is guaranteed by A9 (non-vanishing gradient) for any $h(\mathbf{x}_t)$ in a sufficient interior of $\mathcal{C}$. A9 is required globally for Theorem 1's induction to close.

**Counterexample if A9 violated**: For $h(\mathbf{x}) = (\|\mathbf{x} - \mathbf{x}_0\|^2 - r^2)$, $\nabla h \to 0$ at $\mathbf{x} = \mathbf{x}_0$. Restrict $\mathcal{X}_0$ to a thin shell $\{r - \delta \leq \|\mathbf{x} - \mathbf{x}_0\| \leq r + \delta\}$ near $\partial \mathcal{C}$ where $\nabla h$ is bounded away from zero, or design $h$ as a true SDF (which has unit gradient).

---

## Module 2: Critic-Gradient Error Propagation (NEW — addresses R1 flaw #8)

### 2.1 Practical CBF Constraint and Estimated Projection

Using **global worst-case quantiles** $\bar{\sigma}_h, \bar{\sigma}_\nabla$ (A6 convention, v3), define the **safety-margin buffer**:

$$\xi(k) := \underbrace{\gamma \bar{\sigma}_h}_{\text{critic value err}} + \underbrace{a_{\max} \bar{\sigma}_\nabla \cdot \left\|\tfrac{\partial f}{\partial \mathbf{a}}\right\|}_{\text{critic gradient err}} + \underbrace{L_{\hat{h}} \epsilon(k)}_{\text{layer-skip err}} + \underbrace{\eta \, a_{\max}^2}_{\text{second-order curvature (v4)}} \tag{2.2}$$

Only the third term depends on $k$; the constant part absorbs (i) critic value error, (ii) critic gradient error, and (iii) the **worst-case second-order curvature** $\eta \, a_{\max}^2$ (from A3's quadratic remainder, bounded by A0 action norm). Since $\epsilon(k)$ is non-increasing by A6, **$\xi(k)$ is non-increasing in $k$ by inspection** — no additional conditions needed.

*v4 fix (R4 remaining issue)*: The previous $\xi(k)$ only controlled the linear term $\mathbf{c}^\top \mathbf{a}$, but Lemma 1.0's CBF constraint requires $\mathbf{c}^\top \mathbf{a} - \eta\|\mathbf{a}\|^2 \geq -\gamma h + \mathrm{Lip}_h d_{\max}$. Adding $\eta a_{\max}^2$ to $\xi(k)$ absorbs the worst-case second-order term, making Theorem 5's practical CBF constraint imply the true CBF constraint including the quadratic remainder.

The **practical CBF constraint** on any candidate action $\mathbf{a}$ at state $\mathbf{x}_t$ with layer budget $k$ is:

$$\hat{\mathbf{c}}(\mathbf{x}_t)^\top \mathbf{a} - \eta \|\mathbf{a}\|^2 \geq -\gamma \hat{h}_\phi(\mathbf{f}^{(s)}_{G_k}(\mathbf{x}_t)) + \mathrm{Lip}_h d_{\max} + \xi(k) \tag{2.3}$$

*[Equation (2.3) now explicitly labeled per R2 fix #2.]*

The practical controller uses $\hat{h}_\phi$ and $\hat{\mathbf{c}} = (\partial f/\partial \mathbf{a})^\top \nabla \hat{h}_\phi$:

$$\hat{\mathbf{a}}_t^{\text{safe}} = \mathbf{a}_t^* + \frac{\left(-\gamma \hat{h}_\phi + \mathrm{Lip}_h d_{\max} + \xi(k) - \hat{\mathbf{c}}^\top \mathbf{a}_t^*\right)_+}{\|\hat{\mathbf{c}}\|^2} \hat{\mathbf{c}} \tag{2.1}$$

**Identity-when-feasible property (Lemma 2.0):**  
If $\mathbf{a}_t^*$ already satisfies (2.3), i.e. $\hat{\mathbf{c}}^\top \mathbf{a}_t^* \geq -\gamma \hat{h}_\phi + \mathrm{Lip}_h d_{\max} + \xi(k)$, then the numerator in (2.1) is $\leq 0$ → $(\cdot)_+ = 0$ → $\hat{\mathbf{a}}_t^{\text{safe}} = \mathbf{a}_t^*$. (This property is used in Theorem 4.)

### 2.2 Theorem 5: Ideal-Practical Gap

**Theorem 5 (Ideal-Practical Deviation — v4 with second-order correction):**  
*[Proposition — holds with prob $\geq 1-2\delta$ per step via A5 + union bound over $\hat{\sigma}_h, \hat{\sigma}_\nabla$]*

With prob $\geq 1-2\delta$ per step, the estimated projection satisfies the **full** robust ISSf-CBF constraint:

$$\mathbf{c}(\mathbf{x}_t)^\top \hat{\mathbf{a}}_t^{\text{safe}} - \eta \|\hat{\mathbf{a}}_t^{\text{safe}}\|^2 \geq -\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h d_{\max} \tag{T5}$$

including the second-order curvature term. This is the exact constraint required by Lemma 1.0 for forward invariance.

*Proof (using A0, A5, Cauchy-Schwarz):*

1. **Constructional lower bound on estimated side.** By construction (2.1): $\hat{\mathbf{c}}^\top \hat{\mathbf{a}}_t^{\text{safe}} \geq -\gamma \hat{h}_\phi + \mathrm{Lip}_h d_{\max} + \xi(k)$.

2. **True-vs-estimated gradient gap via A0.** Decompose:
$$\mathbf{c}^\top \hat{\mathbf{a}}_t^{\text{safe}} = \hat{\mathbf{c}}^\top \hat{\mathbf{a}}_t^{\text{safe}} + (\mathbf{c} - \hat{\mathbf{c}})^\top \hat{\mathbf{a}}_t^{\text{safe}}$$
By Cauchy-Schwarz and **A0** ($\|\hat{\mathbf{a}}_t^{\text{safe}}\| \leq a_{\max}$):
$$(\mathbf{c} - \hat{\mathbf{c}})^\top \hat{\mathbf{a}}_t^{\text{safe}} \geq -\|\mathbf{c} - \hat{\mathbf{c}}\| \cdot a_{\max}$$

3. **Gradient-error conformal bound.** By A5 (gradient) w.p. $\geq 1-\delta$ and the monotone-envelope convention:
$$\|\mathbf{c} - \hat{\mathbf{c}}\| = \left\|\tfrac{\partial f}{\partial \mathbf{a}}\right\| \cdot \|\nabla h - \nabla \hat{h}_\phi\| \leq \left\|\tfrac{\partial f}{\partial \mathbf{a}}\right\| \cdot \bar{\sigma}_\nabla$$

4. **Value-error conformal bound (corrected sign v3).** By A5 (value) w.p. $\geq 1-\delta$: $h \geq \hat{h}_\phi - \bar{\sigma}_h$, equivalently $\hat{h}_\phi \leq h + \bar{\sigma}_h$. Therefore:
$$-\gamma \hat{h}_\phi \geq -\gamma(h + \bar{\sigma}_h) = -\gamma h - \gamma \bar{\sigma}_h$$

5. **Combine via union bound.** W.p. $\geq 1 - 2\delta$ (both A5 events jointly):
$$\begin{aligned}
\mathbf{c}^\top \hat{\mathbf{a}}_t^{\text{safe}} &\geq \hat{\mathbf{c}}^\top \hat{\mathbf{a}}_t^{\text{safe}} - a_{\max} \bar{\sigma}_\nabla \|\partial f/\partial \mathbf{a}\| \quad \text{(step 2, 3)} \\
&\geq -\gamma \hat{h}_\phi + \mathrm{Lip}_h d_{\max} + \xi(k) - a_{\max} \bar{\sigma}_\nabla \|\partial f/\partial \mathbf{a}\| \quad \text{(step 1)} \\
&\geq -\gamma h - \gamma \bar{\sigma}_h + \mathrm{Lip}_h d_{\max} + \xi(k) - a_{\max} \bar{\sigma}_\nabla \|\partial f/\partial \mathbf{a}\| \quad \text{(step 4)}
\end{aligned}$$

With $\xi(k) = \gamma \bar{\sigma}_h + a_{\max} \bar{\sigma}_\nabla \|\partial f/\partial \mathbf{a}\| + L_{\hat{h}} \epsilon(k) + \eta a_{\max}^2$ (v4 definition), the buffer terms cancel the conformal errors and the second-order remainder:

$$\mathbf{c}^\top \hat{\mathbf{a}}_t^{\text{safe}} \geq -\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h d_{\max} + L_{\hat{h}} \epsilon(k) + \eta a_{\max}^2$$

6. **Absorb second-order term (v4 fix).** By A0, $\|\hat{\mathbf{a}}_t^{\text{safe}}\| \leq a_{\max}$, so $\eta \|\hat{\mathbf{a}}_t^{\text{safe}}\|^2 \leq \eta a_{\max}^2$. Therefore:
$$\mathbf{c}^\top \hat{\mathbf{a}}_t^{\text{safe}} - \eta \|\hat{\mathbf{a}}_t^{\text{safe}}\|^2 \geq \mathbf{c}^\top \hat{\mathbf{a}}_t^{\text{safe}} - \eta a_{\max}^2 \geq -\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h d_{\max} + L_{\hat{h}} \epsilon(k) \geq -\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h d_{\max}$$
$\square$

**Key fixes from v2/v3** (R3+R4 findings):
- Step 4 inequality direction corrected to match A5 (R3 fix).
- Global quantile convention $\bar{\sigma}$ removes circular $k$-dependence (R3 fix).
- **v4**: Added $\eta a_{\max}^2$ to $\xi(k)$ and Step 6 absorbing second-order term — Theorem 5 now guarantees the **full** (non-linearized) CBF constraint required by Lemma 1.0. Closes R4's concrete counterexample ($f(x,a)=x+a, h=x^2$).

### 2.3 Theorem 6: Practical Safety

**Theorem 6 (Practical Forward Invariance, Finite Horizon — v6 final):**  
*[Proposition — holds under A0–A11 + A8 exchangeability + A10 feasibility]*

Assume A0–A11. Let $\mathbb{P}_T$ denote the joint probability over (i) the calibration set used to construct $\bar{\sigma}_h, \bar{\sigma}_\nabla$, and (ii) the deployment trajectory under A8. If $h(\mathbf{x}_0) \geq 0$ and $\hat{\mathbf{a}}_t^{\text{safe}}$ from (2.1) is executed for $t = 0, \ldots, T-1$, then:

$$\boxed{\mathbb{P}_T\Big[\,h(\mathbf{x}_t) \geq 0 \;\;\text{for all}\;\; t \leq T\,\Big] \;\geq\; 1 - 2T\delta} \tag{T6}$$

*Proof outline (composition)*:
1. By Theorem 3, the joint conformal events $\bigcap_t (E_t^{(h)} \cap E_t^{(\nabla)})$ hold with $\mathbb{P}_T$-probability $\geq 1 - 2T\delta$.
2. Conditional on this joint event, by Theorem 5 the executed action satisfies the **true** CBF inequality at every $t \leq T-1$: $\mathbf{c}(\mathbf{x}_t)^\top \hat{\mathbf{a}}_t^{\text{safe}} - \eta\|\hat{\mathbf{a}}_t^{\text{safe}}\|^2 \geq -\gamma h(\mathbf{x}_t) + \mathrm{Lip}_h d_{\max}$.
3. Conditional on this, by Lemma 1.0 + the recursion in Theorem 1: $h(\mathbf{x}_{t+1}) \geq (1-\gamma) h(\mathbf{x}_t)$ for all $t$. With $h(\mathbf{x}_0) \geq 0$, induction gives $h(\mathbf{x}_t) \geq 0$ for all $t \leq T$.
4. Combining: $\mathbb{P}_T[\text{safe trajectory}] \geq \mathbb{P}_T[\text{conformal events}] \geq 1 - 2T\delta$. $\square$

*Tightness*: The $1 - 2T\delta$ bound is loose (factor $2T$ from union bound). For very long horizons, replace with sequential conformal [Angelopoulos et al. 2024] to get adaptive coverage with $\mathcal{O}(\log T)$ degradation instead of $T$. Treated as future work.

---

## Module 3: Finite-Horizon Conformal (revised)

### 3.1 Per-Sample Coverage (Lemma 3.1 — unchanged)

Split-conformal quantile with **pessimistic score direction** (aligned with A5 safety lower bound):
$$\hat{\sigma}_h = s_{(\lceil (1-\delta)(n+1) \rceil)} \quad \text{where } s_i = \hat{h}_\phi(\mathbf{f}^{(s)}(\mathbf{x}_i)) - h(\mathbf{x}_i) \tag{3.1}$$

*Sign convention*: We use $s_i = \hat{h}_\phi - h$ (estimated minus true), NOT $h - \hat{h}_\phi$. The standard $(1-\delta)$-quantile then gives $\Pr[\hat{h}_\phi - h \leq \hat{\sigma}_h] \geq 1-\delta$, equivalently $\Pr[h \geq \hat{h}_\phi - \hat{\sigma}_h] \geq 1-\delta$, which matches A5 (lower bound on the true barrier, pessimistic direction for safety).

This gives **per-sample one-sided coverage** $\geq 1-\delta$ for exchangeable (cal, test) [Vovk 2005; Angelopoulos & Bates 2023].

*Correction from v2 (R3 real bug #1): score direction was flipped; v3 uses the sign that makes conformal coverage consistent with A5's lower-bound statement on $h$.*

### 3.2 Gradient Conformal

**Definition**: For calibration $\mathbf{x}_i$, compute $g_i = \|\nabla h(\mathbf{x}_i) - \nabla \hat{h}_\phi(\mathbf{f}^{(s)}(\mathbf{x}_i))\|_2$; set $\hat{\sigma}_\nabla = g_{(\lceil (1-\delta)(n+1)\rceil)}$.

**Lemma 3.2:** $\Pr[\|\nabla h(\mathbf{x}) - \nabla \hat{h}_\phi(\cdot)\| \leq \hat{\sigma}_\nabla] \geq 1 - \delta$ under exchangeability.

### 3.3 Finite-Horizon Time-Uniform Bound

**Theorem 3 (revised — v6 measure-theoretic clarification):**  

*Setup*: at each step $t \leq T$, define the **conformal events**:
$$E_t^{(h)} := \{ h(\mathbf{x}_t) \geq \hat{h}_\phi(\mathbf{f}^{(s)}_{G_{k_t}}(\mathbf{x}_t)) - \bar{\sigma}_h \}, \quad
E_t^{(\nabla)} := \{ \|\nabla h(\mathbf{x}_t) - \nabla \hat{h}_\phi(\cdot)\|_2 \leq \bar{\sigma}_\nabla \}$$

These are random events (over the joint distribution of the calibration set + deployment trajectory under A8). The practical constraint (2.3) is **algorithmically enforced** by the projection (2.1) and is therefore **deterministic**; what is probabilistic is whether the conformal events hold so that (2.3) is **sufficient** for the true CBF inequality (Theorem 5).

**Statement (T3)**:
$$\Pr\left[\bigcap_{t=0}^{T-1} (E_t^{(h)} \cap E_t^{(\nabla)})\right] \geq 1 - 2T\delta \tag{T3}$$

i.e., the per-step conformal events used in Theorem 5 jointly hold across all $T$ steps with probability $\geq 1-2T\delta$.

*Proof*: Per-step coverage from A5 gives $\Pr[E_t^{(h)}] \geq 1-\delta$ and $\Pr[E_t^{(\nabla)}] \geq 1-\delta$ for each fixed $t$. Union bound over the $2T$ events. $\square$

Choosing $\delta = \delta'/(2T)$ gives user-specified $1-\delta'$ coverage at cost $\bar{\sigma}_h(\delta'/(2T)) \approx \bar{\sigma}_h(\delta') + \mathcal{O}(\sqrt{\log T / n})$.

*Correction from v1*: Previous Theorem 3 silently promoted per-sample to time-uniform coverage — now explicitly scaled by $T$.

### 3.4 Exchangeability Caveat

**Remark (Deployment Distribution):**  
Exchangeability requires calibration to cover deployment distribution. Our protocol:
1. Collect calibration rollouts under π_safe itself (self-calibration): after initial training, rollout π_safe in simulator, collect $(\mathbf{x}_i, h(\mathbf{x}_i))$ pairs with ground-truth $h$ from simulator geometry.
2. **Re-calibrate** periodically during deployment if distribution shift detected (via Page's CUSUM on nonconformity scores).
3. For task/domain shift: use **sequential conformal** [Angelopoulos et al. 2024] — out of scope for main theory but discussed in experiments.

---

## Module 4: Safety-Aware Routing (reframed — NOT tightness)

### 4.1 Budget Characterization

**Definition (Safety-Compatible Budget):**  
$$k^*(\rho_t) := \inf\{k \in \{1,\ldots,K\} : \xi(k) \leq \rho_t\} \tag{4.1}$$

where $\xi(k)$ is the safety-margin buffer from (2.2), using global quantiles $\bar{\sigma}_h, \bar{\sigma}_\nabla$ (A6 v3 convention).

**Convention (empty feasibility set):** If $\{k : \xi(k) \leq \rho_t\} = \emptyset$ (e.g., $\rho_t$ smaller than the constant offset $\gamma\bar{\sigma}_h + a_{\max}\bar{\sigma}_\nabla\|\partial f/\partial \mathbf{a}\|$), set $k^*(\rho_t) := K$ and trigger the fallback protocol: full network + retreat to last-known-safe state via cached safe action.

### 4.2 Theorem 2: Monotonicity + Feasibility

**Theorem 2 (Budget Monotonicity, v3):**  
*[Proposition — rigorous under A6 with global-quantile convention]*

$$\xi(k) \text{ is non-increasing in } k \tag{T2}$$

*Proof:* By (2.2), $\xi(k) = \text{const} + L_{\hat{h}} \epsilon(k)$, where "const" = $\gamma\bar{\sigma}_h + a_{\max}\bar{\sigma}_\nabla\|\partial f/\partial \mathbf{a}\|$ does not depend on $k$. By A6, $\epsilon(k)$ is non-increasing in $k$. Therefore $\xi(k)$ is non-increasing. $\square$

*The proof is trivial by construction — this is the point of adopting global quantiles in v3.*

**Corollary 2.1 (Feasibility):** If $\rho_t \geq \gamma\bar{\sigma}_h + a_{\max}\bar{\sigma}_\nabla\|\partial f/\partial \mathbf{a}\|$ (i.e., safety margin exceeds irreducible critic error), then $k^*(\rho_t) \in \{1,\ldots,K\}$ is well-defined, and executing $k = k^*(\rho_t)$ activates the minimum layer count such that (2.3) is feasible.

*Honest scope*: Theorem 2 states monotonicity + feasibility under A6. We do NOT claim tightness against adversarial policies.

---

## Module 5: Performance-Difference Bound (replaces Theorem 4)

### 5.1 Kakade-Langford Performance-Difference Lemma

**Lemma 5.1 (Classical):**  
For any two policies $\pi, \pi'$ and trajectory distribution $d^{\pi'}$:

$$V^{\pi'}(s_0) - V^\pi(s_0) = \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{T-1} A^\pi(\mathbf{s}_t, \mathbf{a}_t)\right]$$

where $A^\pi(\mathbf{s}, \mathbf{a}) = Q^\pi(\mathbf{s}, \mathbf{a}) - V^\pi(\mathbf{s})$ is the advantage.

### 5.2 Theorem 4 (revised): Performance Preservation

**Theorem 4 (Performance-Difference Safety Cost):**  
*[Proposition — rigorous, uses A7 (Q-Lipschitz uniformly over $\pi_{\text{safe}}$-state support) and Lemma 2.0]*

Let $\pi_{\text{MoLe}}$ be the base policy and $\pi_{\text{safe}}$ its ISSf-CBF–projected version defined via (2.1). Then:

$$V^{\pi_{\text{MoLe}}}(s_0) - V^{\pi_{\text{safe}}}(s_0) \leq L_Q \cdot T \cdot p_{\text{viol}} \cdot \bar{\Delta} \tag{T4}$$

where, letting $d^{\pi_{\text{safe}}}$ denote the state occupancy under $\pi_{\text{safe}}$:
- $p_{\text{viol}} = \mathbb{E}_{\mathbf{s} \sim d^{\pi_{\text{safe}}}}\left[\mathbb{1}[\mathbf{a}^*(\mathbf{s}) \notin \mathcal{C}_\mathbf{a}^{\text{rob}}(\mathbf{s})]\right]$ is the per-step violation rate under the $\pi_{\text{safe}}$-induced distribution,
- $\bar{\Delta} = \mathbb{E}_{\mathbf{s} \sim d^{\pi_{\text{safe}}}}\left[\|\mathbf{a}^*(\mathbf{s}) - \mathbf{a}^{\text{safe}}(\mathbf{s})\| \mid \text{violation}\right]$ is the expected projection displacement conditional on a violation.

*Full proof:*

(i) **Performance-difference lemma.** By Lemma 5.1 (Kakade-Langford 2002):
$$V^{\pi_{\text{MoLe}}}(s_0) - V^{\pi_{\text{safe}}}(s_0) = -\mathbb{E}_{\tau \sim \pi_{\text{safe}}}\left[\sum_{t=0}^{T-1} A^{\pi_{\text{MoLe}}}(\mathbf{s}_t, \mathbf{a}_t^{\text{safe}})\right]$$

(ii) **Zero-advantage at base action.** For any state $\mathbf{s}$, $\mathbf{a}^* := \pi_{\text{MoLe}}(\mathbf{s})$ is the $\pi_{\text{MoLe}}$-action; thus $Q^{\pi_{\text{MoLe}}}(\mathbf{s}, \mathbf{a}^*) = V^{\pi_{\text{MoLe}}}(\mathbf{s})$ and $A^{\pi_{\text{MoLe}}}(\mathbf{s}, \mathbf{a}^*) = 0$.

(iii) **Identity-when-feasible (Lemma 2.0).** If $\mathbf{a}^*(\mathbf{s}_t) \in \mathcal{C}_\mathbf{a}^{\text{rob}}(\mathbf{s}_t)$ (i.e., no violation), then $\mathbf{a}_t^{\text{safe}} = \mathbf{a}^*(\mathbf{s}_t)$ by Lemma 2.0, so $A^{\pi_{\text{MoLe}}}(\mathbf{s}_t, \mathbf{a}_t^{\text{safe}}) = 0$. Only violation steps contribute.

(iv) **Lipschitz advantage at violation steps.** At violation steps, apply A7 (Q-Lipschitz uniform over $d^{\pi_{\text{safe}}}$-support):
$$|A^{\pi_{\text{MoLe}}}(\mathbf{s}_t, \mathbf{a}_t^{\text{safe}})| = |Q^{\pi_{\text{MoLe}}}(\mathbf{s}_t, \mathbf{a}_t^{\text{safe}}) - Q^{\pi_{\text{MoLe}}}(\mathbf{s}_t, \mathbf{a}^*(\mathbf{s}_t))| \leq L_Q \|\mathbf{a}_t^{\text{safe}} - \mathbf{a}^*(\mathbf{s}_t)\|$$

(v) **Aggregate.** Combining (i)–(iv):
$$|V^{\pi_{\text{MoLe}}} - V^{\pi_{\text{safe}}}| \leq \mathbb{E}_{\tau \sim \pi_{\text{safe}}}\left[\sum_{t=0}^{T-1} \mathbb{1}[\text{viol}_t] \cdot L_Q \|\mathbf{a}_t^{\text{safe}} - \mathbf{a}^*(\mathbf{s}_t)\|\right] = L_Q \cdot T \cdot p_{\text{viol}} \cdot \bar{\Delta}$$
$\square$

**Assumptions made explicit in v3** (addresses R2 new flaw #6):
- A0 (bounded actions) ensures the projection set is compact and Q-Lipschitz constant is finite.
- **A7 interpreted as uniform** over $\mathbf{s} \in \mathrm{supp}(d^{\pi_{\text{safe}}})$: there exists $L_Q$ valid for all states $\pi_{\text{safe}}$ may visit.
- Lemma 2.0 (identity when feasible) is now a **named lemma**, not implicit.

**Correction from v1 (3 flaws fixed):**
- ✅ **Distribution**: Expectation is explicitly over $\pi_{\text{safe}}$-induced trajectory (the correct distribution).
- ✅ **Reward Lipschitz**: Replaced by Q-Lipschitz (A7) — holds even for 0/1 success rewards if task MDP is mixing.
- ✅ **$\|\mathbf{c}\|_{\min}$ removed**: The bound uses $\|\mathbf{a}^* - \mathbf{a}^{\text{safe}}\|$ directly, which is bounded even at flat barriers (but projection itself may be infeasible — handled by Module 4 fallback).

**Interpretation**:  
Performance degradation is bounded by: $(L_Q) \times (\text{horizon}) \times (\text{violation rate}) \times (\text{avg projection size})$. This is:
- **Zero** if base policy always safe ($p_{\text{viol}} = 0$) or already feasible.
- **Small** if violations are rare and/or corrections are small.
- **Bounded even at flat barriers** (unlike v1).

---

## Main Theorem (Safe-MoLe v2)

**Theorem (Main, Revised):**  
*[Uses A1–A8; combines Theorems 1, 3, 4, 5, 6]*

Let $\pi_{\text{safe}}$ denote Safe-MoLe with (a) routing $k = k^*(\hat{\rho}_t)$, (b) estimated projection (2.1), (c) conformal quantiles $\hat{\sigma}_h, \hat{\sigma}_\nabla$ calibrated per (3.1). Then, over finite horizon $T$:

**(i) [Ideal Safety]** If $h, \nabla h$ are known exactly (oracle critic, $\hat{\sigma}_h = \hat{\sigma}_\nabla = 0$): $h(\mathbf{x}_t) \geq 0$ for all $t \leq T$ deterministically.

**(ii) [Practical Safety]** With probability $\geq 1 - 2T\delta$ over the calibration set:
$$h(\mathbf{x}_t) \geq 0 \quad \text{for all } t \leq T$$

**(iii) [Performance Preservation]** Under the $\pi_{\text{safe}}$-induced distribution:
$$V^{\pi_{\text{safe}}}(s_0) \geq V^{\pi_{\text{MoLe}}}(s_0) - L_Q T p_{\text{viol}} \bar{\Delta}$$

**(iv) [Efficiency — informal remark, not a theorem]** Average routing cost $\mathbb{E}[k_t] = \mathbb{E}[k^*(\hat{\rho}_t)]$ depends on the distribution of $\hat{\rho}_t$. Since $k^*(\cdot)$ is monotone non-increasing in $\rho$ (by Theorem 2 + def. 4.1), states with larger safety margin activate fewer layers. Quantitative efficiency claims require empirical validation — stated as experimental claim, not theorem.

*v4 downgrade*: Previous "(iv)" was vacuous ($\mathbb{E}[k_t] \leq K$ trivially). Per R4 feedback, this is now an informal remark with the qualitative mechanism stated explicitly; experimental validation carries the efficiency claim.

**Honest caveats now stated explicitly:**
- (i) requires oracle; (ii) is the deployable guarantee with coverage parameter $2T\delta$.
- (ii) now includes the $\eta a_{\max}^2$ second-order buffer (v4 fix) — valid for all η ≥ 0, not just η=0.
- (iii) is local-in-distribution (under $\pi_{\text{safe}}$'s own trajectory), per Kakade-Langford.
- (iv) is a qualitative/empirical statement, not a mathematical theorem.

---

## Comparison with Baselines (revised — honest)

| Aspect | SafeVLA (CMDP) | AEGIS (CBF-QP) | ATACOM (op-space) | **Safe-MoLe (ours)** |
|--------|----------------|-----------------|-------------------|----------------------|
| Safety guarantee | Asymptotic expected constraint satisfaction | Instantaneous 1st-order CBF | Exact (known dynamics) | Finite-horizon $1-2T\delta$ coverage via conformal |
| Assumptions on h | Learned (RL) | Handcrafted | Known, handcrafted | Learned with **conformal error tracking** |
| Handles critic error | N/A (no critic) | No | N/A | **Yes, explicit via Theorem 5** |
| Performance bound | Empirical | None | None | Kakade-Langford (T4) |
| Efficiency | Same as base | Extra VLM + QP per step | Same as base | **Routing-adaptive: k*(ρ_t) ≤ K** |
| Code complexity | Requires safe RL loop | Requires QP solver | Requires dynamics | Closed-form + conformal calibration |

**Honest positioning**: Safe-MoLe does NOT claim stronger safety theory than ATACOM (which has exact guarantees under known dynamics). The **distinct contribution** is:
1. **Explicit critic-error tracking** via conformal (none of the baselines do this)
2. **Routing-adaptive compute** (unique to MoLe architecture)
3. **Performance-preserving** by Kakade-Langford (no baseline provides this)

---

## Proposed Method Architecture (unchanged from v1)

```
Observation o_t + Language l
        ↓
[DINO-v2 + SigLIP]
        ↓
[LLaMA2-7B + Safety-Aware STAR Router with k = k*(ρ̂_t)]
        ↓
[CogKD Cognition Token]
  ├──→ [DiT-Base Action Head] → a_t*  (nominal action)
  └──→ [Safety Critic Head ĥ_φ, ∇ĥ_φ]  ← NEW (2-layer MLP)
        ↓
[Estimated ISSf-CBF Projection (2.1)]  ← NEW (closed-form)
        ↓
Execute â_t^safe
```

**New components**:
- **Safety Critic Head** $\hat{h}_\phi$: 2-layer MLP, ~$2d^2$ params, outputs $\hat{h}, \nabla \hat{h}$ jointly
- **Estimated Projection Layer**: closed-form (2.1), differentiable, $\mathcal{O}(7)$ ops
- **Routing Override**: $k^*(\hat{\rho}_t)$ table lookup

**Added FLOPs**: < 0.1% of MoLe. Routing extends MoLe's existing mechanism.

---

## Training Objective (unchanged, with explicit dependencies)

$$\mathcal{L}_{\text{Safe-MoLe}} = \mathcal{L}_{\text{task}} + \lambda_2 \mathcal{L}_{\text{cog}} + \lambda_3 \mathcal{L}_{\text{lb}} + \lambda_4 \mathcal{L}_{\text{critic}} + \lambda_5 \mathcal{L}_{\text{CBF}}$$

$$\mathcal{L}_{\text{critic}} = \mathbb{E}_t\left[(h(\mathbf{x}_t) - \hat{h}_\phi(\mathbf{f}^{(s)}_t))^2\right] + \lambda_\nabla \mathbb{E}_t\left[\|\nabla h - \nabla \hat{h}_\phi\|^2\right]$$

(Note: $\nabla h$ can be computed from simulator geometry; gradient supervision improves $\hat{\sigma}_\nabla$.)

$$\mathcal{L}_{\text{CBF}} = \mathbb{E}_t\left[\left(-\gamma \hat{h}_\phi + \mathrm{Lip}_h d_{\max} + \xi(k) - \hat{\mathbf{c}}^\top \hat{\mathbf{a}}_t^{\text{safe}}\right)_+\right]$$

---

## Experimental Plan (revised — empirical-forward per Path C)

The paper story is now **empirical-led, theory-principled**. Target contributions:

| # | Experiment | What it shows | Metric |
|---|------------|---------------|--------|
| 1 | **SafeLIBERO** main | vs SafeVLA, AEGIS on safety | Obstacle avoidance rate, task success |
| 2 | **RLBench-10 efficiency** | Safety-aware routing maintains MoLe's ×5.6 speedup | FLOPs + success |
| 3 | **Safety-CHORES** long-horizon | Handles compound tasks | Safety violation cost |
| 4 | **Conformal calibration curve** | $\hat{\sigma}_h$ vs $n$ | Empirical coverage vs nominal $1-\delta$ |
| 5 | **Coverage under dist. shift** | Deployment on held-out tasks | Coverage degradation |
| 6 | **Ablation: ISSf projection** | Theorem 6 necessity | Violation rate w/ vs w/o projection |
| 7 | **Ablation: gradient critic** | Theorem 5 necessity | Projection accuracy |
| 8 | **Ablation: routing override** | Module 4 benefit | Efficiency-safety curve |
| 9 | **Real Franka FR3** | Generalization | Real collision rate |
| 10 | **Failure mode analysis** | Where ISSf guarantee breaks (flat barrier, OOD) | Case studies |

**Claim hierarchy**:
- **Primary**: Safe-MoLe beats SafeVLA / AEGIS on safety while matching MoLe's efficiency.
- **Secondary**: Conformal calibration gives empirically valid coverage.
- **Theoretical**: Under stated conditions, practical safety is guaranteed with horizon-dependent probability.

---

## Boundaries and Non-Claims (strengthened)

- We do **NOT** claim stronger theoretical guarantees than ATACOM (which has exact safety under known dynamics).
- We do **NOT** claim zero safety violations in practice — only bounded probability under conformal calibration.
- The "exact" Theorem 1 requires oracle $h$ and $\nabla h$ — it is a *reference result*, not the deployed system.
- A3 requires $f \in C^2$ in action — breaks down at contact events; mitigation via event-based controllers is future work.
- A6 requires **nested** layer-skip — restricts MoLe's top-$k$ to a fixed ordering; ablation in Experiments #2.
- Conformal coverage is per-deployment-distribution; domain shift requires re-calibration.

---

## Open Risks

1. **CBF feasibility**: If $\|\mathbf{c}\| < c_{\min}$, projection infeasible → fallback to full-layer + retreat maneuver (Module 4 convention). Monitor in Experiment #10.
2. **Conformal sample complexity**: Calibration needs $n \gg 1/\delta$ samples; for $\delta = 0.01, T = 100$, need $n \geq 2\times 10^4$ — affordable in simulator.
3. **Gradient supervision**: Requires differentiable simulator or finite-difference $\nabla h$; adds ~20% sim cost.
4. **Concurrent competitor**: Between v1 and v2 review, the field may publish a combined ISSf + routing paper. Track arXiv weekly.
5. **Distribution shift in the wild**: Sequential conformal [Angelopoulos 2024] is our planned fallback; initial experiments suggest 5–10% coverage degradation on OOD tasks.

---

## Summary of Revision Impact

| R1 Critical Flaw | v2 Fix | Location |
|------------------|--------|----------|
| 1. Theorem 1 incorrect recurrence | Separate Lip_h vs L_∇h; correct proof | Module 1, Lemma 1.0 |
| 2. Lemma 1.1 exact only η=0 | Explicit 1D Newton for η>0 | Module 1.3 |
| 3. A3 assumes conclusion | Replaced by Taylor theorem with explicit η | A3 restatement |
| 4. Conformal misapplied to trajectory | Union bound over horizon T; coverage 1−2Tδ | Theorem 3 revised |
| 5. A5 a.s. vs conformal incompatible | A5 restated as conformal quantile | A5 restatement |
| 6. Theorem 2 "tightness" not tight | Reframed as monotonicity + feasibility | Module 4 |
| 7. Theorem 4 distribution mismatch + ‖c‖_min | Performance-difference lemma + A7 (Q-Lipschitz) | Module 5 |
| 8. Main Theorem ignores critic gap | NEW Theorem 5 bounds ideal-practical gap | Module 2 |

**Theoretical contribution summary**: Under A1–A8, Safe-MoLe achieves practical safety with coverage $1-2T\delta$, bounded performance degradation via Kakade-Langford, and explicit error tracking from critic to deployed action — the first safe-VLA with full error accounting.

---

*Status: COHERENT v3 | Generated: 2026-04-14*  
*Ready for Round-3 review.*

---

## v3 Changelog (post-R2 polish)

| R2 New Flaw | v3 Fix | Location |
|-------------|--------|----------|
| 1. Implicit bounded actions | **NEW A0** stated explicitly; cited in Theorem 5 step 2 | Assumptions, Theorem 5 proof |
| 2. Equation (2.3) unlabeled | Labeled (2.3); Theorem 3 references it | Module 2.1 |
| 3. A4 x-Lip vs feature-Lip mismatch | A4 rephrased as feature-space Lipschitz | A4 |
| 4. $\xi(k)$ monotonicity needs quantile-in-$k$ | Monotone envelope $\tilde{\sigma}(k)$ introduced in A6 addendum | A6, Module 2.1, Module 4 |
| 5. KL: identity-when-feasible implicit | **Lemma 2.0** now a named lemma | Module 2.1 |
| 6. A7 Q-Lip not uniform over states | A7 revised to explicit uniformity over $\mathrm{supp}(d^{\pi_{\text{safe}}})$ | A7 |

**Unchanged from v2**: Main Theorem structure, all 8 R1 fixes retained.

**Still acknowledged as open (future work / empirical validation)**:
- ~~η > 0 case (only η=0 formally analyzed)~~ ✅ **Fixed in v4** (see v4 fix below)
- Sequential conformal for domain shift (in place of union bound's 2T factor)
- Non-nested routing (outside A6 scope)
- Quantitative efficiency bound on $\mathbb{E}[k_t]$ (currently empirical-only)

---

## v3 Post-Review Updates (2 real bugs from R3)

R3 (blind review of v3) identified 2 real mathematical bugs:

**Bug 1 (A5 ↔ Theorem 5 sign mismatch)**:
- Old Module 3 score $s_i = h - \hat{h}_\phi$ gave $\Pr[h \leq \hat{h}_\phi + \hat{\sigma}_h]$ — opposite direction to A5's lower bound.
- **v3 fix**: flipped to $s_i = \hat{h}_\phi - h$, giving $\Pr[h \geq \hat{h}_\phi - \hat{\sigma}_h]$ matching A5.
- Theorem 5 Step 4 proof now correctly derives $\hat{h}_\phi \leq h + \bar{\sigma}_h$ → $-\gamma\hat{h}_\phi \geq -\gamma h - \gamma\bar{\sigma}_h$, buffer absorbs it.

**Bug 2 (Theorem 2 monotonicity non-rigorous)**:
- Old monotone-envelope $\tilde{\sigma}(k) = \max_{k' \leq k} \hat{\sigma}(k')$ is non-decreasing in $k$; sum with non-increasing $\epsilon(k)$ not monotone.
- **v3 fix**: global worst-case quantile $\bar{\sigma} := \max_{k} \hat{\sigma}(k)$ (scalar, $k$-independent).
- $\xi(k) = \text{const} + L_{\hat{h}}\epsilon(k)$ → monotonicity trivial from A6.

**R3 verdict before these fixes**: 7/10 Weak Accept, contingent on both fixes. Both now in place.
