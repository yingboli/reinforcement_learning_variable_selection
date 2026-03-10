# Variable Selection: Literature Review & Simulation Design Notes

A short scan of recent variable selection work in strong journals and typical simulation setups used there. Focus is on regression/linear models and designs you can adapt for RL-based variable selection.

---

## 1. Recent Papers in Strong Outlets

### Bayesian Analysis

- **Bayesian Variable Selection Under High-Dimensional Settings with Grouped Covariates** (2025, Agarwal et al.)  
  High-dimensional regression with **grouped, correlated covariates**; not all variables in a group need to be relevant. Good reference for **group structure** and **within-group correlation** in simulations.

- **A Variational Spike-and-Slab Approach for Group Variable Selection** (2025, Ge, Lin & Liu)  
  **Spike-and-slab** for linear regression with **grouped variables**; variational inference. Useful for **group-sparse** and **correlated-block** designs.

- **Bayesian Variable Selection in a Million Dimensions** (2023, Jankowiak, AISTATS)  
  MCMC for very large *p* (e.g. genomics) with **sublinear cost per iteration**; GLMs (Gaussian, binomial, negative binomial). Reference for **scaling (n, p)** and **high-dimensional** regimes.

### Empirical / Method Comparison

- **High-dimensional regression in practice: an empirical study of finite-sample prediction, variable selection and ranking** (Wang et al., *Statistics and Computing* / PMC 7026376)  
  Large comparison over **2,300+ scenarios** (synthetic and semi-synthetic). Varies **n, p, sparsity, signal strength, multicollinearity**. Separates **prediction**, **variable selection**, and **ranking**. Often cited for **benchmark n/p and correlation setups**.

- **High-Dimensional Regression Under Correlated Design: An Extensive Simulation Study** (Springer chapter)  
  Focus on **correlated design**; compares Lasso, SCAD, LARS, etc. Uses **Toeplitz and block correlation** structures.

### Reinforcement Learning & Feature Selection

- **Efficient Variable Selection Using Reinforcement Learning for Big Data** (OpenReview; REVS)  
  **MDP + RL** for variable selection; **online policy iteration / TD**. Emphasizes **high-dimensional, correlated** features.

- **Knockoff-Guided Feature Selection via A Single Pre-trained Reinforced Agent** (2024, arXiv)  
  **Knockoffs + deep Q-network**; supervised/unsupervised selection. Good for **FDR/knockoff-style** evaluation.

- **Sequential knockoffs for variable selection in reinforcement learning** (LSE; SEEK)  
  **Minimal sufficient state** in MDPs; selection consistency; applied to sepsis (MIMIC-III). Reference for **RL + variable selection** and **sequential/state** settings.

---

## 2. Common Simulation Design Elements

From the papers above, typical factors and ranges:

| Factor | Typical choices | Notes |
|--------|------------------|--------|
| **Sample size (n)** | 100, 200, 500, 1000 | Sometimes 50–200 for “hard” high-dim; 500–2000 for consistency |
| **Number of predictors (p)** | 50, 100, 500, 1000, 4000 | Often **p > n** or **p ≈ n** for “high-dimensional” |
| **Sparsity (s₀ or n_informative)** | 5, 10, 20, or 10–20% of p | True model size |
| **Signal strength / SNR** | SNR = 0.5, 1, 2, 4; or β in [0.5, 2] | Often scale β so target R² or SNR is fixed |
| **Correlation structure** | Independence, Toeplitz ρ^|i−j|, block diagonal | ρ = 0.3, 0.5, 0.7, 0.9 common |
| **Error distribution** | N(0, σ²) | σ chosen to hit target SNR or R² |
| **Covariate distribution** | N(0, Σ), sometimes standardized | Σ from correlation structure above |

### Correlation Structures (Widely Used)

1. **Toeplitz (AR-like)**  
   - Σ_{ij} = ρ^{|i−j|}, ρ ∈ (0, 1).  
   - Easy to implement; one parameter; mimics “ordered” (e.g. time/location) predictors.

2. **Block diagonal**  
   - Variables in blocks of size **p_block**; within-block correlation ρ; between-block 0.  
   - Good for **grouped** or **cluster** structure; can put signals inside or across blocks.

3. **Compound symmetry**  
   - All pairs have correlation ρ.  
   - Simple “everything correlated” case; often harder for selection.

4. **Semi-synthetic**  
   - **Real X** (e.g. from a public dataset), **synthetic y** from a known linear model plus noise.  
   - Realistic X; known ground truth for selection and prediction.

---

## 3. Suggested Simulation Designs (Aligned with Literature)

### Design A: “Classic” high-dimensional (Wang-style)

- **n** = 200, **p** = 500 (or 200, 1000).  
- **s₀** = 10 or 20 (true signals).  
- **X**: N(0, Σ), Σ = Toeplitz with ρ = 0.5 (or 0.3, 0.7).  
- **β**: s₀ non-zero coefficients; e.g. equal (e.g. 1) or random signs × constant.  
- **σ**: set so **SNR = Var(Xβ)/σ²** = 1, 2, or 4 (or target **R²**).  
- **Replicates**: 100–500 per (n, p, ρ, SNR) cell.

Good for: comparing RL vs Lasso/Adaptive Lasso/RFE/MCMC under **correlated design** and **known truth**.

### Design B: Vary difficulty (sparsity + correlation)

- Fix **n** = 200, **p** = 100.  
- **Sparsity**: s₀ ∈ {5, 10, 20, 40}.  
- **Correlation**: Toeplitz ρ ∈ {0, 0.3, 0.5, 0.7}.  
- **SNR** ∈ {1, 2, 4}.  
- One “standard” β pattern (e.g. first s₀ coefficients non-zero, rest 0).

Shows how **precision/recall/F1** and **MSE/R²** change with **sparsity** and **correlation**.

### Design C: Grouped / block structure (Bayesian Analysis–style)

- **n** = 200, **p** = 100; **block size** = 10 (10 blocks).  
- **Within-block correlation** ρ = 0.6; **between-block** = 0.  
- **True model**: 2–3 blocks with non-zero effects (e.g. one predictor per active block, or block-sparse).  
- **σ** to get SNR ≈ 2.

Useful if you later add **group-aware** baselines or **group structure** to the RL setup.

### Design D: Scaling (n and p)

- **Small**: n = 100, p = 50; **medium**: n = 200, p = 200; **large**: n = 500, p = 1000.  
- **s₀** ∝ √p or 10% of p (e.g. 5, 14, 32).  
- Toeplitz ρ = 0.5; SNR = 2.  
- Compare **runtime** and **selection accuracy** (e.g. F1, support recovery) across methods.

### Design E: Semi-synthetic (real X, synthetic y)

- Take **real X** from a public dataset (e.g. diabetes, Boston, or a genomics/ML dataset with p ≥ 20).  
- Choose **true support** S (e.g. 5–10 indices).  
- **y** = X_S β_S + ε, ε ∼ N(0, σ²); set **σ** for target R² (e.g. 0.5, 0.7).  
- Many replicates with **random S** or **random β** (fix S).

Improves **realism** and is aligned with “real covariates, simulated response” benchmarks.

---

## 4. Performance Metrics (From Literature)

- **Variable selection**: Precision, Recall, F1, support recovery (0/1), Hamming distance.  
- **Prediction**: MSE, R² (or RMSE) on held-out test set.  
- **Uncertainty / calibration** (if applicable): coverage of intervals, FDR (e.g. for knockoff-style methods).  
- **Computation**: runtime, number of iterations (for MCMC/RL).

Your current **precision, recall, F1, test R², test MSE** are in line with these.

---

## 5. References (Short List)

1. Wang et al., “High-dimensional regression in practice…” (PMC7026376 / *Statistics and Computing*).  
2. Bayesian Analysis (2025): Agarwal et al. (grouped covariates); Ge, Lin & Liu (spike-and-slab groups).  
3. Jankowiak, “Bayesian Variable Selection in a Million Dimensions,” AISTATS 2023.  
4. “High-Dimensional Regression Under Correlated Design,” Springer chapter (Toeplitz/block designs).  
5. REVS (OpenReview), Knockoff-guided RL (arXiv 2403.04015), SEEK (LSE) for RL-based selection.

---

## 6. Quick Implementation Tips for Your Codebase

- **Toeplitz Σ**: `scipy.linalg.toeplitz(r)` with `r = [1, ρ, ρ², …, ρ^{p−1}]`; then `X = Z @ L` where `L` is Cholesky of Σ, Z iid N(0,1).  
- **Block correlation**: block-diagonal matrix with each block `(1−ρ)I + ρ11'`; same Cholesky sampling.  
- **SNR**: if `β` and `X` are fixed, `σ² = Var(Xβ) / SNR`; for random X, approximate Var(Xβ) or target **R²** and set σ accordingly.  
- **Replicates**: 100–500 per scenario; report **mean ± std** (or median and IQR) for F1, MSE, R², runtime.

If you tell me which design (A–E) you want first, I can help translate it into concrete `n_samples`, `n_informative`, correlation matrix, and `noise` (or SNR) in your `run_simulation` / data-generation setup.
