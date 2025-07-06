# Kolmogorov-Complexity × Free-Energy Active Inference

---

### Abstract

We propose a strict information-theoretic test of Friston’s Free-Energy Principle (FEP). An agent that learns a generative model of its sensorimotor stream should, in principle, **shorten the algorithmic description length of that stream**. We formalise this as an upper-bound Kolmogorov complexity objective and prove that, under mild regularity conditions, the *per-step decrease* in that bound is proportional to the *per-step decrease* in variational free energy. The result yields a single scalar yard-stick—**bits per time-step**—linking perception, action and learning, and exposes an efficiency ceiling unreachable by policies that merely optimise extrinsic reward.

---

## 1 Problem Statement

> Can an actively sampling agent compress its own sensory history **better than universal lossless compressors**, and does every saved bit correspond to a drop in free energy?

If true, the FEP gains a falsifiable, byte-level prediction; if false, the principle’s universality claim weakens.

---

## 2 Theoretical Foundations

### 2.1 Kolmogorov Complexity

For a string $s$, $K(s)$ is the length (in bits) of the shortest program that outputs $s$ on a fixed universal Turing machine and halts. $K(\,\cdot)$ is uncomputable; agents can only achieve **upper bounds**.

*Two-part code bound* for a parametric model $\theta$:

$$
\widehat{K}_\theta(o_{1:T}) \;=\; \underbrace{\| \theta \|_2^2 \log 2}_{\text{model description}} \;+\; \sum_{t=1}^{T}\!-\log_2 p_\theta(o_t \mid h_{<t})
\tag{1}
$$

### 2.2 Variational Free Energy

Given latent posterior $q(z_{1:T})$ and generative model $p_\theta$,

$$
F_T \;=\; \underbrace{ \mathbb{E}_{q}\!\big[-\log p_\theta(o_{1:T}\!\mid\!z_{1:T})\big] }_{\text{accuracy}} \;+\; \underbrace{ \mathrm{KL}\!\big[q(z_{1:T}) \,\|\, p_\theta(z_{1:T})\big] }_{\text{complexity}}
\tag{2}
$$

Active inference selects actions that minimise an *expected* $F_{t+\tau}$.

### 2.3 Resource-Bounded Complexity

We adopt Levin’s **Kt** measure:

$$
K_t(s) \;=\; \min_{\pi\;\text{halts in}\;t} \big\{\, |\pi| \;+\; \log t \big\}
\tag{3}
$$

capturing both brevity and computation time, matching biological energy limits.

---

## 3 Compression–Free-Energy Coupling

**Proposition 1 (Bit-for-bit equivalence).**
For any model class in which $q(z_t)$ is conditionally independent of $o_{>t}$ given $o_{\le t}$, the expected one-step change satisfies

$$
\underbrace{\mathbb{E}\!\left[\Delta \widehat{K}_\theta(o_{1:T})\right]}_{\text{compression gain}}
\;=\;
(1+\varepsilon_T)\,
\underbrace{\mathbb{E}\!\left[\Delta F_T\right]}_{\text{free-energy drop}}
\tag{4}
$$

where $|\varepsilon_T| \le \tfrac{1}{T}$ under bounded posterior entropy.

*Sketch.* Rearranging (1) and (2) shows identical NLL terms; model-size penalty and KL differ by at most $O(1)$ bits per episode, vanishing asymptotically.

---

## 4 Efficiency Metrics

| Symbol                                          | Meaning                                  | Desired trend  |
| ----------------------------------------------- | ---------------------------------------- | -------------- |
| $G_T = K_{\text{LZMA}} - \widehat{K}_\theta$    | Compression gap vs. universal compressor | non-decreasing |
| $\rho_T = \tfrac{\mathrm{d}F_T}{\mathrm{d}G_T}$ | Bit-elasticity of free energy            | converge to −1 |
| $\eta_T = \tfrac{G_T}{E_T}$                     | Bits saved per Joule or CPU-cycle        | maximise       |

A perfectly efficient agent reaches $G_T > 0$ while keeping $|\rho_T - 1| < \delta$ for small $\delta$.

---

## 5 Predictions & Tests

1. **Monotone Compression:** $G_T$ must grow (or plateau) as learning progresses; any sustained decline falsifies model sufficiency.
2. **Tight Coupling:** Pearson correlation between $G_T$ and $-F_T$ should exceed 0.9 once transients fade.
3. **Baseline Supremacy:** For stationary environments, there exists $T^\*$ such that $\widehat{K}_\theta(o_{1:T^\*}) \lt K_{\text{LZMA}}(o_{1:T^\*})$.

Failure of any single prediction invalidates the claim that active inference yields algorithmically meaningful compression.

---

## 6 Broader Implications

* **Unified Complexity Yard-stick** — Replaces heuristic entropy or likelihood metrics with a foundation tied to algorithmic information theory.
* **Resource-Bounded Rationality** — Kt injects explicit computational cost, making the theory testable on real hardware or biological energy budgets.
* **Model Selection via Description Length** — Favours generative architectures that truly shorten code length rather than merely over-fitting likelihood.
* **Objective Benchmark for FEP** — Moves debate from philosophical narratives to byte-level, peer-verifiable experiments.

---

### Conclusion

Binding Kolmogorov complexity and variational free energy along a single bit-axis yields the clearest, falsifiable formulation of the Free-Energy Principle to date. An agent that cannot demonstrably compress its own history beyond universal compressors has no claim to “minimising surprise”; one that can, supplies the missing algorithmic proof.


---

## Running the Example

1. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install numpy
   ```
2. Run the toy compression script:
   ```bash
   python src/main.py
   ```
   The script simulates a binary sensor stream, fits a Bernoulli model by maximum likelihood and reports the predicted metrics `G_T` and `\rho_T` alongside the LZMA baseline.
