# Kolmogorov x Free-Energy PoC

We propose a theoretical bridge between **Kolmogorov complexity** (KC) and **Friston’s variational free‑energy principle** (FEP). Both frameworks quantify “surprise” but from distinct vantage points: KC measures the shortest description of a data stream, while FEP bounds the negative log evidence of an agent’s sensory exchanges.  
We show:

* (i) For any sensory trajectory \(o_{1:T}\), *resource‑bounded* KC is upper‑bounded by cumulative free‑energy plus a model‑size constant.  
* (ii) An active‑inference agent that minimizes free‑energy therefore performs an *implicit compression search*; the asymptotic free‑energy gap to zero offers a computable estimate of how far the agent stands from the (uncomputable) true KC.  
* (iii) Efficiency constraints appear naturally by moving from ideal KC to Levin’s \(K_t\) (program length + log runtime) and by adding an inverse‑temperature multiplier to the FEP’s complexity term.  

The union yields a single scalar objective in **bits per timestep**, simultaneously governing perception, action, and memory cost.  

---

### Background Notation  

| Symbol | Meaning |
|--------|---------|
| \(o_t\) | Observation at discrete time \(t\) |
| \(a_t\) | Action issued by the agent |
| \(\theta\) | Parameters of the agent’s generative model \(p_\theta(o_{1:T}, s_{1:T})\) |
| \(q_\phi(s_{1:T})\) | Variational posterior over latent states |
| \(K(o_{1:T})\) | True Kolmogorov complexity of the sensory tape |
| \(K_t(o_{1:T}) = \min_{p\!: p(o)=o_{1:T}}\,|p| + \log_2\,\mathrm{run\_time}(p)\) | Levin’s speed‑prior KC |
| \(F_t\) | Step‑wise free energy |
| \(\mathcal L_\theta\) | Description length afforded by the agent’s model |
 
---

