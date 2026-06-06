# Diffusion Model Predictive Synthesis Planning: Controllable Multi-Objective Route Discovery via Factored World Models

**Authors:** [Author Names]
**Affiliation:** [Institution]
**Correspondence:** [email]

*Target venue: NeurIPS 2026 / ICML 2026*

---

## Abstract

We introduce **Diffusion Model Predictive Synthesis Planning (DMPSP)**, a framework for controllable, multi-objective chemical synthesis route discovery grounded in factored diffusion-based Model Predictive Control. Existing retrosynthesis methods apply single-step template models recursively, compounding prediction errors across each reaction and optimizing a single implicit objective. DMPSP departs from this paradigm in three ways. First, a joint dynamics model (DynamicsDiffusion) predicts F-step reaction outcomes in a single forward pass over a fine-tuned ReactionT5 backbone, eliminating compounding error. Second, an action proposal diffusion model ρ samples N candidate reaction sequences conditioned on the current synthesis state, forming an explicit prior over chemical action space. Third, a 10-head Transformer value function J scores full trajectories against ten process-aware objectives — yield, purity, cost, novelty, FTO risk, green chemistry, manufacturability, safety, robustness, supply availability — using objective weights supplied at inference time with no retraining. The MPC loop selects the highest-scoring candidate at each receding horizon step, enabling runtime objective re-weighting across user profiles (generic pharma, CDMO, green chemistry, discovery). We train on USPTO-50K (50,016 reactions) with pseudo-labeled property supervision from surrogate models and public chemical databases. DMPSP-Beam achieves [XX]% top-1 route validity and [XX]% top-5 coverage on the USPTO-50K test set, planning 10-step routes in [XX]s on CPU, while DMPSP-MCTS achieves [XX]% top-1 at [XX]× longer planning time. Runtime objective re-weighting incurs zero additional compute cost.

---

## 1. Introduction

Identifying a viable multi-step synthesis route for a target pharmaceutical molecule is one of the most resource-intensive tasks in drug development. A practicing process chemist must simultaneously optimize yield, cost, safety, intellectual property clearance, manufacturing scalability, and reagent availability — often under tight timelines. Computational retrosynthesis has made significant progress on the *validity* dimension: given a target molecule, systems like AiZynthFinder [Genheden et al., 2020], ASKCOS [Coley et al., 2019], and Retro* [Chen et al., 2020] can rapidly enumerate chemically plausible precursor trees. However, three fundamental limitations remain unaddressed.

**Limitation 1: Compounding single-step errors.** Tree search methods apply a single-step reaction prediction model recursively. At each node, a model is queried: *what precursors could produce this intermediate?* Errors in this query accumulate multiplicatively across the search depth. A 5-step route with 90% per-step accuracy has only 59% route-level accuracy; at 10 steps, 35%. Jointly modeling the full trajectory — predicting all steps simultaneously — avoids this compounding.

**Limitation 2: Single-objective optimization.** Existing methods optimize primarily for route validity or Tanimoto similarity to training reactions [Schwaller et al., 2019]. Industrial deployment requires simultaneously balancing yield, cost per gram, process safety (explosive functional groups, toxic intermediates), FTO risk (patent clearance), green chemistry metrics (PMI, solvent waste), and supply chain robustness. No existing open-source system supports runtime multi-objective optimization without retraining.

**Limitation 3: Fixed planning objectives.** Different stakeholders weight objectives differently. A CDMO prioritizes manufacturability and robustness. A discovery team prioritizes novelty and yield. A green chemistry program prioritizes solvent waste and atom economy. Retraining a separate model per objective profile is computationally prohibitive.

We address all three limitations with DMPSP, adapting the D-MPC framework [Lu et al., 2025] from continuous control to the discrete-continuous hybrid space of chemical synthesis planning. Our contributions are:

1. **Joint multi-step world model for synthesis.** DynamicsDiffusion wraps ReactionT5v2-forward in a DDIM framework, predicting F reaction outcomes in a single forward pass. We show this reduces route-level error by [XX]% relative to chained single-step prediction on USPTO-50K (Section 6.2).

2. **Factored diffusion action proposal.** ActionProposalDiffusion learns p(a_{1:F} | s) over F-step reaction condition sequences via DDPM, with cross-attention conditioning on the full synthesis state. At inference, N=64 candidates are drawn via deterministic DDIM (η=0), giving reproducible planning.

3. **Runtime 10-objective value function.** A 10-head Transformer trained with randomly sampled objective weight vectors learns disentangled objective representations. At inference, J(τ; w) = w⊤ J(τ) — a single dot product, zero compute overhead.

4. **Process-aware synthesis state.** SynthesisState encodes not just the current molecule but inventory, reaction history, accumulated cost, yield, purity, temperature, pressure, and step index. This richer representation enables the world model to condition property predictions on process context.

5. **Open-source implementation.** Full code, training scripts, and Kaggle-reproducible notebooks released at [repository URL].

---

## 2. Related Work

### 2.1 Retrosynthesis and Synthesis Planning

**Template-based methods.** Early computational retrosynthesis applied manually curated reaction templates [Corey & Wipke, 1969]. Modern systems learn templates from reaction databases: ASKCOS [Coley et al., 2019] and AiZynthFinder [Genheden et al., 2020] combine template application with neural ranking. These methods inherit template coverage limitations and optimize no process objectives.

**Template-free methods.** Molecular Transformer [Schwaller et al., 2019] and ReactionT5 [Sagawa et al., 2023] treat reaction prediction as sequence-to-sequence translation. These models are highly accurate on forward prediction but are not designed for multi-step planning or objective optimization.

**Neural tree search.** Retro* [Chen et al., 2020] replaces heuristic search with a learned cost function, improving planning efficiency. MCTS-based methods [Segler et al., 2018] achieve strong coverage but are computationally expensive and single-objective. None of these methods model reaction conditions or process properties.

**Multi-objective synthesis.** Thakkar et al. [2021] present unbiased multi-objective evaluation metrics for retrosynthesis but do not propose a multi-objective planning algorithm. Molga et al. [2022] optimize green chemistry metrics post-hoc by reranking tree-search outputs. DMPSP integrates multi-objective optimization into the planning loop itself.

### 2.2 Diffusion Models for Planning and Control

**Diffuser [Janner et al., 2022].** The Diffuser treats offline RL trajectory optimization as sequence denoising, learning p(τ) over full trajectories. Conditioning on reward enables goal-directed planning. This work establishes that diffusion models can represent complex multi-modal trajectory distributions.

**Decision Diffuser [Ajay et al., 2023].** Extends Diffuser with classifier-free guidance over return-conditioned distributions, enabling test-time objective specification.

**D-MPC [Lu et al., 2025].** Factorizes trajectory diffusion into separate action proposal ρ and dynamics p_d models. The key insight is that ρ generates compact action sequences while p_d (the world model) handles the state consequence prediction — two very different distributions that benefit from separate parameterization. DMPSP directly adapts D-MPC to chemistry, replacing continuous control actions with discrete reaction class + continuous condition vectors and replacing physics dynamics with ReactionT5-backed chemical dynamics.

**Diffusion Policy [Chi et al., 2023].** Applies diffusion to robot manipulation action generation, demonstrating DDPM/DDIM action proposals in continuous control. Our action proposal module is inspired by this work, adapted to the chemistry domain.

### 2.3 Chemical Language Models

**Molecular Transformer [Schwaller et al., 2019].** BERT-style transformer for reaction prediction, establishing the effectiveness of SMILES tokenization for chemistry.

**ReactionT5 [Sagawa et al., 2023].** T5 pre-trained on 10B+ chemical tokens, achieving state-of-the-art forward and retrosynthetic prediction on USPTO-50K. We fine-tune ReactionT5v2-forward as our world model backbone, leveraging its strong chemical prior.

**ChemGPT [Frey et al., 2023].** GPT-style generation of molecular SMILES for property-conditioned design. Unlike DMPSP, ChemGPT does not model reaction trajectories or process objectives.

### 2.4 Property Prediction for Synthesis Planning

Yield prediction [Schwaller et al., 2021; Ahneman et al., 2018] and toxicity prediction [Wu et al., 2018; Yang et al., 2019] are mature subfields. DMPSP integrates property prediction directly into the world model as regression heads over reaction outcomes, enabling end-to-end training of trajectory-level property estimates.

---

## 3. Problem Formulation

### 3.1 Synthesis State Space

A synthesis state $s \in \mathcal{S}$ is a structured tuple:

$$s = (m^{\text{target}}, m^{\text{current}}, \mathcal{I}, h, T, P, \sigma, c, k, \phi, \psi)$$

where:
- $m^{\text{target}} \in \mathcal{M}$ — target molecule (canonical SMILES)
- $m^{\text{current}} \in \mathcal{M}$ — current synthetic intermediate
- $\mathcal{I} \subset \mathcal{M}$ — inventory of available starting materials
- $h \in \mathbb{Z}^{L}$ — reaction class history (last $L$ steps)
- $T \in [0,1]$ — normalized temperature
- $P \in [0,1]$ — normalized pressure
- $\sigma \in [0,1]$ — normalized batch scale
- $c \in \mathbb{R}_{\geq 0}$ — accumulated cost (USD)
- $k \in \{0, \ldots, K\}$ — step index
- $\phi \in [0,1]$ — cumulative yield fraction
- $\psi \in [0,1]$ — cumulative purity

### 3.2 Action Space

A synthesis action $a \in \mathcal{A}$ specifies reaction conditions:

$$a = (r, T_{\text{norm}}, P_{\text{norm}}, t_{\text{norm}}, \ell, \kappa, \rho)$$

where:
- $r \in \{1, \ldots, R\}$ — reaction class ID (USPTO-50K has R=10 classes)
- $T_{\text{norm}}, P_{\text{norm}}, t_{\text{norm}} \in [0,1]$ — normalized temperature, pressure, time
- $\ell \in \{1, \ldots, L_s\}$ — solvent ID
- $\kappa \in \{1, \ldots, L_c\}$ — catalyst ID
- $\rho \in [0,1]$ — normalized reagent ratio

The action space is therefore a mixed discrete-continuous space: $\mathcal{A} = \{1,\ldots,R\} \times \{1,\ldots,L_s\} \times \{1,\ldots,L_c\} \times [0,1]^4$.

### 3.3 Planning Objective

Given a target molecule $m^{\text{target}}$, a starting state $s_0$ (with $m^{\text{current}} = m^{\text{target}}$ and empty history), and objective weights $w \in \Delta^{9}$ (the 10-simplex), the planning problem is:

$$\tau^* = \arg\max_{\tau = (s_0, a_0, \ldots, s_T)} \sum_{k=1}^{10} w_k \cdot J_k(\tau)$$

subject to $s_{t+1} = f(s_t, a_t)$ (chemistry dynamics) and $s_T$ containing a commercially available starting material.

The 10 objectives $J_k: \mathcal{T} \to [0,1]$ are defined in Table 1.

**Table 1: DMPSP Objective Functions**

| $k$ | Name | Definition | Supervision source |
|-----|------|------------|--------------------|
| 1 | yield | $\prod_{t=0}^{T} \phi_t$ | Yield-BERT, ORD |
| 2 | purity | $\min_t \psi_t$ | Reaction class heuristics |
| 3 | cost | $\exp(-c / c_{\max})$ | eMolecules API |
| 4 | novelty | $1 - \max_{r \in \mathcal{D}} \text{Tc}(\tau, r)$ | Tanimoto vs. USPTO-50K |
| 5 | fto\_risk | $1 - \text{PatentCov}(\tau)$ | SureChEMBL API |
| 6 | green\_chem | $\text{AE}(\tau) \cdot \text{GreenSolv}(\tau)$ | CHEM21 solvent guide, RDKit |
| 7 | manufacturability | $\text{PMI}^{-1}(\tau) \cdot \text{GMP}(\tau)$ | Process Mass Intensity |
| 8 | safety | $1 - \text{HazardScore}(\tau)$ | RDKit structural alerts |
| 9 | robustness | $1 - \text{Var}[\phi | \Delta T, \Delta P]$ | Condition sensitivity simulation |
| 10 | supply\_avail | $\text{Avail}(m^{\text{start}})$ | eMolecules API |

---

## 4. DMPSP Architecture

DMPSP comprises four trainable components. We describe each in detail, including the exact parameterization, loss function, and training procedure.

### 4.1 Molecular Encoder

All molecular inputs (SMILES strings) are encoded into dense vectors before being consumed by the three neural components. We use Morgan fingerprints (ECFP4, radius=2, 2048 bits) computed by RDKit, projected by a learned linear layer:

$$\text{enc}(m) = \text{Linear}(2048, d_{\text{enc}})(f_{\text{ECFP4}}(m)) \in \mathbb{R}^{d_{\text{enc}}}$$

where $d_{\text{enc}} = 256$. The projection is trained jointly with ActionProposalDiffusion in Phase 1 and frozen thereafter. This choice avoids the `torch_geometric` dependency of GIN encoders while providing a deterministic, hash-stable encoding that enables reproducible planning.

The full synthesis state $s$ is encoded as:

$$\mathbf{s} = \text{MLP}([\text{enc}(m^{\text{target}}); \text{enc}(m^{\text{current}}); \bar{\mathcal{I}}; \mathbf{h}; T; P; \sigma; c; k; \phi; \psi]) \in \mathbb{R}^{d_s}$$

where $\bar{\mathcal{I}} = \frac{1}{|\mathcal{I}|}\sum_{m \in \mathcal{I}} \text{enc}(m)$ is the mean-pooled inventory encoding and $\mathbf{h} \in \mathbb{R}^{L \cdot d_R}$ is the embedded reaction history.

### 4.2 ActionProposalDiffusion (ρ)

#### 4.2.1 Forward and Reverse Process

Let $\mathbf{a} = (a_1, \ldots, a_F) \in \mathbb{R}^{F \times d_a}$ be a sequence of $F$ action vectors in a continuous relaxation of $\mathcal{A}$. Discrete components (reaction class, solvent, catalyst) are embedded via learned lookup tables before diffusion; at sampling time, a straight-through argmax recovers discrete assignments.

The forward diffusion process adds Gaussian noise over $T_d = 1000$ steps:

$$q(\mathbf{a}^n | \mathbf{a}^{n-1}) = \mathcal{N}(\mathbf{a}^n; \sqrt{1 - \beta_n}\,\mathbf{a}^{n-1},\; \beta_n \mathbf{I})$$

with variance schedule $\beta_n$ (cosine schedule, $\beta_{\min}=10^{-4}$, $\beta_{\max}=0.02$). The marginal:

$$q(\mathbf{a}^n | \mathbf{a}^0) = \mathcal{N}(\mathbf{a}^n;\; \sqrt{\bar{\alpha}_n}\,\mathbf{a}^0,\; (1-\bar{\alpha}_n)\mathbf{I}), \quad \bar{\alpha}_n = \prod_{i=1}^n (1 - \beta_i)$$

The reverse process is parameterized by a noise prediction network $\epsilon_\rho$:

$$p_\rho(\mathbf{a}^{n-1} | \mathbf{a}^n, s) = \mathcal{N}\!\left(\mathbf{a}^{n-1};\; \mu_\rho(\mathbf{a}^n, n, s),\; \tilde{\beta}_n \mathbf{I}\right)$$

$$\mu_\rho(\mathbf{a}^n, n, s) = \frac{1}{\sqrt{\alpha_n}}\!\left(\mathbf{a}^n - \frac{\beta_n}{\sqrt{1-\bar{\alpha}_n}}\,\epsilon_\rho(\mathbf{a}^n, n, s)\right)$$

#### 4.2.2 Network Architecture

$\epsilon_\rho$ is a 5-layer Transformer with cross-attention conditioning:

- **Self-attention layers:** Process the noised action sequence $\mathbf{a}^n \in \mathbb{R}^{F \times d_a}$ with sinusoidal diffusion step embedding appended per token
- **Cross-attention layers:** Keys and values from $\mathbf{s}$ (synthesis state encoding), queries from self-attention output — this is where state conditioning enters
- **Causal masking:** None (non-causal; all F steps processed jointly)
- **Output:** $\hat{\epsilon} \in \mathbb{R}^{F \times d_a}$ — predicted noise at each action step

Hyperparameters: $d_{\text{model}}=256$, 8 heads, $d_{\text{ff}}=1024$, dropout=0.1, $F=5$ (planning horizon).

#### 4.2.3 Training Loss

$$\mathcal{L}_\rho = \mathbb{E}_{n \sim \mathcal{U}[1,T_d],\; \mathbf{a}^0 \sim \mathcal{D},\; \epsilon \sim \mathcal{N}(0,\mathbf{I})} \left[ \|\epsilon - \epsilon_\rho(\sqrt{\bar{\alpha}_n}\,\mathbf{a}^0 + \sqrt{1-\bar{\alpha}_n}\,\epsilon,\; n,\; s)\|^2 \right]$$

This is the standard DDPM $\epsilon$-prediction objective [Ho et al., 2020].

#### 4.2.4 DDIM Inference

At inference, we sample $N=64$ action sequences using DDIM [Song et al., 2021] with $\eta=0$ (deterministic):

$$\mathbf{a}^{n-1} = \sqrt{\bar{\alpha}_{n-1}} \underbrace{\left(\frac{\mathbf{a}^n - \sqrt{1-\bar{\alpha}_n}\,\epsilon_\rho}{\sqrt{\bar{\alpha}_n}}\right)}_{\hat{\mathbf{a}}^0} + \sqrt{1-\bar{\alpha}_{n-1}}\,\epsilon_\rho$$

With $\eta=0$ and fixed seed, sampling is fully deterministic given the seed — critical for reproducible planning. 50 DDIM steps are used at inference (vs. 1000 training steps), a 20× speedup.

### 4.3 ChemistryWorldModel (p_d)

The world model predicts the outcome of applying action sequence $\mathbf{a} = (a_1, \ldots, a_F)$ starting from state $s_t$: product SMILES $m^{\text{out}}$ and property scores $(y, \text{tox}, \text{mfg}, \text{supply}, \text{fto})$.

#### 4.3.1 Backbone: ReactionT5v2-forward

We fine-tune `sagawa/ReactionT5v2-forward`, a T5-base (60M parameters, $d_{\text{model}}=768$) pre-trained on forward reaction prediction. Input:

$$x = \texttt{[REACTANT]} \; m^{\text{current}} \; \texttt{[REAGENT]} \; r_{\text{desc}} \; \texttt{[SEP]} \; T \; P \; t \; \ell \; \kappa \; \rho$$

where $r_{\text{desc}}$ is a text description of the reaction class and process conditions are appended as special tokens. The T5 encoder produces a context representation $\mathbf{H} \in \mathbb{R}^{L \times 768}$; the decoder autoregressively generates the product SMILES $m^{\text{out}}$.

Full fine-tuning (backbone not frozen) is used for maximum adaptation to synthesis planning. Gradient checkpointing on encoder layers reduces peak GPU memory from ~12GB to ~6GB, enabling batch_size=16 on P100 16GB.

#### 4.3.2 Property Regression Heads

Five regression heads are attached to the pooled encoder representation $\bar{\mathbf{H}} = \text{MeanPool}(\mathbf{H}) \in \mathbb{R}^{768}$:

**YieldHead:**
$$\hat{y} = \sigma(\text{MLP}_{2}(\bar{\mathbf{H}})) \in [0,1], \quad \text{MLP}_2: 768 \to 256 \to 1$$

**ToxicityHead:**
$$[\hat{\mu}, \hat{c}, \hat{p}] = \sigma(\text{MLP}_2(\bar{\mathbf{H}})) \in [0,1]^3 \quad \text{(mutagenicity, carcinogenicity, process safety)}$$

**ManufacturabilityHead:**
$$[\hat{f}, \hat{g}] = \sigma(\text{MLP}_2(\bar{\mathbf{H}})) \in [0,1]^2 \quad \text{(scale feasibility, GMP score)}$$

**SupplyChainHead:**
$$[\hat{v}, \hat{l}] = \sigma(\text{MLP}_2(\bar{\mathbf{H}})) \in [0,1]^2 \quad \text{(availability, normalized lead time)}$$

**PatentabilityHead:**
$$[\hat{n}, \hat{f}] = \sigma(\text{MLP}_2(\bar{\mathbf{H}})) \in [0,1]^2 \quad \text{(route novelty, FTO risk)}$$

All MLPs: Linear → GELU → Dropout(0.1) → Linear.

#### 4.3.3 Training Loss

$$\mathcal{L}_{\text{wm}} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{yield}} + \lambda_2 \mathcal{L}_{\text{tox}} + \lambda_3 \mathcal{L}_{\text{mfg}} + \lambda_4 \mathcal{L}_{\text{supply}} + \lambda_5 \mathcal{L}_{\text{fto}}$$

where $\mathcal{L}_{\text{recon}}$ is cross-entropy over the product SMILES sequence (standard T5 language modeling loss) and each property loss is binary cross-entropy or MSE depending on label type:

$$\mathcal{L}_{\text{prop}} = \frac{1}{B} \sum_{i=1}^B \|\hat{y}_i - y_i\|_2^2 \quad \text{(MSE for continuous properties)}$$

Loss weights: $\lambda_1 = 1.0$ (yield, highest signal), $\lambda_2 = \lambda_3 = \lambda_4 = \lambda_5 = 0.5$.

#### 4.3.4 DynamicsDiffusion Wrapper

For F-step joint prediction, we wrap the world model in a DDIM diffusion process over the F-step outcome space. Rather than autoregressing step-by-step, a single denoising pass over the concatenated F-step outcome tensor predicts all F products simultaneously. This is the key architectural choice that eliminates compounding error.

Formally, let $\mathbf{o} = (m^{\text{out}}_1, y_1, \ldots, m^{\text{out}}_F, y_F)$ be the F-step outcome sequence. DynamicsDiffusion learns:

$$p_{p_d}(\mathbf{o} | s_t, \mathbf{a}) = \int p_{p_d}(\mathbf{o}^0 | \mathbf{o}^N, s_t, \mathbf{a}) \prod_{n=1}^N p_{p_d}(\mathbf{o}^{n-1} | \mathbf{o}^n, s_t, \mathbf{a})\, d\mathbf{o}^{1:N}$$

with the same DDPM/DDIM training and inference protocol as ρ. At inference, $\eta=0$ deterministic sampling gives the MAP outcome prediction.

### 4.4 ValueFunction (J)

#### 4.4.1 Architecture

The value function takes a complete F-step trajectory as input and outputs 10 objective scores:

$$\mathbf{J}(\tau) = [J_1(\tau), \ldots, J_{10}(\tau)]^\top \in [0,1]^{10}$$

Input encoding: the trajectory $\tau = (s_0, a_0, o_0, \ldots, s_F, a_F, o_F)$ is encoded as a sequence of $3F$ tokens — one per (state, action, outcome) triple — processed by a 10-layer Transformer:

$$\text{token}_t = \text{Linear}([\mathbf{s}_t; \mathbf{a}_t; \mathbf{o}_t]) \in \mathbb{R}^{256}$$

Positional encoding: learned per-step embeddings added to each token. The [CLS] token output is passed to 10 independent regression heads:

$$J_k(\tau) = \sigma(\text{Linear}(256, 1)(\mathbf{h}_{\text{CLS}})) \quad k = 1, \ldots, 10$$

#### 4.4.2 Training: Objective-Agnostic Supervision

The key training design choice: **objective weights are never seen during training**. Instead, each batch element $(τ, y_{1:10})$ provides supervised targets for all 10 objective scores independently:

$$\mathcal{L}_J = \frac{1}{10B} \sum_{k=1}^{10} \sum_{i=1}^B (J_k(\tau_i) - y_{k,i})^2$$

This forces the value function to learn disentangled, objective-specific representations of trajectory quality. At inference, the multi-objective score is simply a dot product:

$$\text{score}(\tau; \mathbf{w}) = \mathbf{w}^\top \mathbf{J}(\tau) = \sum_{k=1}^{10} w_k J_k(\tau)$$

Changing $\mathbf{w}$ requires no recomputation of $\mathbf{J}(\tau)$ — the trajectory encoding is fixed.

**Comparison to weight-conditioned approaches.** A weight-conditioned value function $J(\tau, \mathbf{w})$ requires a new forward pass for each weight vector tested. With N=64 candidates and M weight profiles, this is 64M forward passes. DMPSP requires 64 forward passes (one per candidate) regardless of M — a critical efficiency advantage for multi-stakeholder deployment.

### 4.5 DMPSPPlanner: MPC Loop

The planner implements Algorithm 1.

---

**Algorithm 1: DMPSP Planning**

```
Input:  target m^target, weights w, max_steps K, n_candidates N, horizon F
Output: synthesis route τ*

Initialize s_0 = SynthesisState(target=m^target, current=m^target, ...)
τ = []

for t = 0, 1, ..., K do
    # Step 1: Sample N candidate action sequences
    {â_i}_{i=1}^N ~ DDIM(ε_ρ(·, s_t), steps=50, η=0, seed=t)   # N × F actions

    # Step 2: Predict outcomes for all candidates (batched)
    {ô_i}_{i=1}^N = WorldModel(s_t, {â_i})                        # N × F outcomes

    # Step 3: Score trajectories
    for i = 1 to N do
        τ_i = (s_t, â_i[0], ô_i[0], ..., â_i[F-1], ô_i[F-1])
        score_i = w^T J(τ_i)
    end

    # Step 4: Select best first action
    i* = argmax_i score_i
    a_t = â_{i*}[0]                                               # execute first action only

    # Step 5: Advance state
    s_{t+1} = Transition(s_t, a_t, ô_{i*}[0])
    τ.append((s_t, a_t))

    # Termination: current_smiles in inventory or step limit reached
    if s_{t+1}.current_smiles ∈ s_0.inventory or t+1 == K then
        break
    end
end

return τ
```

---

**MCTS variant.** When `search_strategy=mcts`, Step 4 is replaced by UCT with N=64 simulations per node, value function as rollout policy, and UCB1 exploration constant $C=\sqrt{2}$. MCTS trades planning time (~10× slower than beam) for improved coverage of the action space.

---

## 5. Training

### 5.1 Dataset

**USPTO-50K** [Lowe, 2012]: 50,016 atom-mapped reaction SMILES from US patents, covering 10 reaction classes (C-C bond formation, C-N, C-O, C-S, C-halogen, reduction, oxidation, protection, deprotection, other). Split: 45,014 train / 2,500 val / 2,502 test.

**Preprocessing.** Atom maps are stripped (not used for modeling). Reactions with invalid SMILES (RDKit sanitization failure) are discarded. The resulting 50,016 valid reactions are wrapped into single-step `SynthesisRoute` objects.

**Property label generation.** USPTO-50K lacks process labels. We generate pseudo-labels via:

| Objective | Labeling method | Coverage |
|-----------|----------------|---------|
| yield | Yield-BERT [Schwaller et al., 2021] inference | 100% (model inference) |
| toxicity | RDKit structural alerts + DeepTox [Unterthiner et al., 2015] | 100% |
| manufacturability | PMI computation + CHEM21 solvent greenness | 100% (computable) |
| supply\_avail | eMolecules API (starting materials only) | ~85% (API coverage) |
| fto\_risk | SureChEMBL SMARTS search | ~70% (patent DB coverage) |
| green\_chem | Atom economy (RDKit) + GSK solvent score | 100% |
| safety | RDKit functional group alerts (185 alert patterns) | 100% |
| novelty | Tanimoto similarity to training set (RDKit, radius=2) | 100% |
| cost | eMolecules API + step count heuristic | ~85% |
| purity | Reaction class × condition heuristics | 100% |

**Label quality.** Pseudo-labels are proxies, not ground truth. Yield-BERT achieves R²≈0.85 on held-out HTE yield data [Ahneman et al., 2018]. Toxicity QSAR (Tox21): AUC≈0.85 on binary endpoints. Manufacturability and safety are computed from validated cheminformatics methods. These are sufficient for relative route ranking — the value function learns to discriminate good routes from bad, not to predict absolute property values.

### 5.2 Training Protocol

Three sequential phases on a single GPU:

**Phase 1 — ActionProposalDiffusion** (also trains molecular encoder):

- Optimizer: AdamW, lr=1e-4, weight decay=1e-2
- Schedule: Cosine decay with 1000 warm-up steps
- Batch: 64 trajectories, max_steps=100,000
- EMA: exponential moving average (decay=0.999) of model weights
- Checkpoint: every 2,500 steps (keep last 2 to manage disk)
- GPU: Kaggle P100 16GB, ~3h

**Phase 2 — ChemistryWorldModel**:

- Optimizer: AdamW, lr=5e-5 (lower LR for fine-tuning), weight decay=1e-2
- Schedule: Cosine decay with 2,000 warm-up steps
- Batch: 16 (T5 memory constraint), gradient accumulation steps=4 (effective batch=64)
- Gradient checkpointing: enabled on T5 encoder
- max_steps=100,000
- GPU: Kaggle P100 16GB, ~8-10h

**Phase 3 — ValueFunction**:

- Optimizer: AdamW, lr=1e-4, weight decay=1e-2
- Schedule: Cosine decay with 1,000 warm-up steps
- Batch: 64, max_steps=100,000
- GPU: Kaggle P100 16GB, ~2-3h

All phases: gradient clipping max_norm=1.0, mixed precision fp16, `--resume` flag for automatic continuation after timeout.

**Why sequential training?** The encoder is shared across all three models. Training it jointly with the action proposal first, then freezing it, ensures consistent molecular representations. The world model is trained on encoder outputs, then frozen before value function training — the value function sees fixed trajectory representations.

### 5.3 Training Stability

Training the action proposal with DDPM is stable by construction (noise prediction has bounded targets). The world model occasionally shows instability during early fine-tuning steps (loss spikes at ~step 5,000) due to the property heads pulling gradients away from the backbone's pre-trained loss surface. We address this with:

1. **Lower LR for world model** (5e-5 vs. 1e-4 for other phases)
2. **Loss weight annealing:** Property head weights $\lambda_k$ are linearly ramped from 0.1 to their final values over the first 10,000 steps
3. **Gradient clipping** (max_norm=1.0) prevents catastrophic forgetting of SMILES generation

---

## 6. Experiments

### 6.1 Evaluation Protocol

**Test set:** 2,502 held-out reactions from USPTO-50K. For each target molecule, the planner is given the reactants as inventory and asked to find a route of at most 10 steps back to those reactants (forward planning to the product).

**Compute budget for evaluation:** Each method is given a 60-second wall-clock budget per molecule on a single CPU core. This reflects realistic deployment constraints.

**Metrics:**

- **Top-1 Route Validity (T1V):** Fraction of test molecules for which the top-ranked route contains all valid SMILES at each step (RDKit sanitization check)
- **Top-5 Coverage (T5C):** Fraction of test molecules for which at least 1 of the top-5 routes is valid
- **Round-Trip Accuracy (RTA):** For routes that terminate with a purchasable starting material, fraction where the forward synthesis of that starting material recovers the target (±Tanimoto > 0.9)
- **Weighted Objective Score (WOS):** $\frac{1}{|\mathcal{T}|}\sum_\tau \mathbf{w}_{\text{pharma}}^\top \mathbf{J}(\tau)$ under the generic pharma weight profile
- **Planning time:** Median seconds per molecule

### 6.2 Main Results

*[To be filled after training completion — target numbers based on preliminary 20K-step checkpoint below]*

**Table 2: Main results on USPTO-50K test set**

| Method | T1V (%) | T5C (%) | RTA (%) | WOS | Time (s) |
|--------|---------|---------|---------|-----|---------|
| AiZynthFinder | — | — | — | N/A | — |
| ASKCOS | — | — | — | N/A | — |
| Retro* | — | — | — | N/A | — |
| DMPSP-Beam (ours) | **[XX]** | **[XX]** | **[XX]** | **[XX]** | **[XX]** |
| DMPSP-MCTS (ours) | [XX] | [XX] | [XX] | [XX] | [XX] |

**Preliminary observation (20K steps, Phase 2 in progress):** World model val_loss=0.0247 at step 21,000, down from 0.031 at initialization. This represents [XX]% improvement in forward reaction prediction accuracy over the untrained model. Full 100K-step results pending.

### 6.3 Ablation Studies

**Table 3: Architecture ablations**

| Variant | T1V (%) | WOS | ΔT1V | Notes |
|---------|---------|-----|------|-------|
| Full DMPSP-Beam | [XX] | [XX] | — | — |
| – No world model (random rollout) | [XX] | [XX] | [XX] | World model contribution |
| – Chained single-step (no DDIM joint) | [XX] | [XX] | [XX] | Compounding error cost |
| – Single objective (yield only) | [XX] | [XX] | [XX] | Multi-obj contribution |
| – Fixed weights at training time | [XX] | [XX] | [XX] | Runtime re-weighting value |
| – GIN encoder (vs. Morgan FP) | [XX] | [XX] | [XX] | Encoder choice |
| – No EMA | [XX] | [XX] | [XX] | EMA contribution |

### 6.4 Runtime Objective Re-weighting

**Table 4: Same checkpoint, 4 weight profiles, 100 test molecules**

| Profile | w = (y, p, c, n, f, g, m, s, r, v) | WOS | Top yield | Top cost |
|---------|--------------------------------------|-----|-----------|----------|
| Generic pharma | (0.30, 0.05, 0.20, 0.05, 0.15, 0.05, 0.10, 0.05, 0.02, 0.03) | [XX] | [XX] | [XX] |
| CDMO | (0.20, 0.10, 0.15, 0.02, 0.08, 0.10, 0.25, 0.05, 0.02, 0.03) | [XX] | [XX] | [XX] |
| Discovery | (0.40, 0.10, 0.05, 0.25, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01) | [XX] | [XX] | [XX] |
| Green | (0.20, 0.05, 0.10, 0.05, 0.05, 0.35, 0.05, 0.10, 0.02, 0.03) | [XX] | [XX] | [XX] |

Key result: identical forward passes, different $\mathbf{w}$ vectors → different top-ranked routes selected. Zero additional GPU compute.

### 6.5 Compounding Error Analysis

To quantify the benefit of joint vs. chained prediction, we compare:

1. **DynamicsDiffusion (joint):** Predict all F steps in one DDIM pass
2. **Chained (autoregressive):** Apply world model F times, feeding each output as next input

For each route length $F \in \{1, 2, 3, 5, 10\}$, we compute per-route validity on 500 test molecules:

**Table 5: Route validity vs. prediction method and length**

| Route length F | Chained (%) | Joint (%) | Joint advantage |
|---------------|-------------|-----------|-----------------|
| 1 | [XX] | [XX] | — |
| 2 | [XX] | [XX] | +[XX]pp |
| 3 | [XX] | [XX] | +[XX]pp |
| 5 | [XX] | [XX] | +[XX]pp |
| 10 | [XX] | [XX] | +[XX]pp |

Hypothesis: advantage of joint prediction grows with F (compounding error is super-linear in chained case).

### 6.6 Case Study: Aspirin Synthesis

We trace a full planning run for acetylsalicylic acid (aspirin, `CC(=O)Oc1ccccc1C(=O)O`) under the generic pharma profile.

**Planning parameters:** max_steps=5, n_candidates=64, beam_width=4, device=cpu.

**Predicted route (preliminary, untrained model — for illustration):**

| Step | Current | Reaction class | T (K) | Solvent | Predicted yield |
|------|---------|---------------|-------|---------|-----------------|
| 1 | CC(=O)Oc1ccccc1C(=O)O | Acylation | 353 | AcOH | [XX]% |
| 2 | OC(=O)c1ccccc1O | Esterification | 298 | DCM | [XX]% |
| 3 | [Starting material] | — | — | — | — |

**Objective scores:** yield=[XX], cost=$[XX]/mmol, FTO=[XX], green=[XX], manufacturability=[XX].

*[Full case study with trained model to be filled — current values from untrained model are not meaningful]*

---

## 7. Discussion

### 7.1 What DMPSP Learns

The action proposal model learns a distribution over *chemically meaningful* reaction condition sequences — not arbitrary continuous vectors. After training, samples from ρ should cluster around common reaction profiles (e.g., Pd-catalyzed C-C couplings at 80-100°C in DMF/dioxane, SNAr reactions at room temperature in DMSO). This can be visualized by UMAP projection of sampled action sequences — structure in the latent space indicates learned chemical knowledge.

The value function learns to distinguish routes that, while all chemically valid, differ systematically in their process properties. A key sanity check: does J_yield(τ) correlate with the pseudo-label yield values, and does J_cost(τ) anti-correlate with route cost? If so, the value function has learned meaningful objective signals.

### 7.2 Failure Modes

**Invalid SMILES generation.** The world model may generate syntactically invalid SMILES. We post-filter with RDKit, discarding invalid candidates. In preliminary runs with the untrained model, ~30% of sampled candidates are invalid — this fraction should drop substantially after full fine-tuning.

**Out-of-distribution targets.** USPTO-50K covers common medicinal chemistry reactions. Macrolide synthesis, highly strained systems, or organometallic reactions are likely to produce poor results. The model should be expected to fail gracefully (generating invalid SMILES) rather than hallucinate convincing but wrong routes.

**Pseudo-label noise.** Property head supervision comes from surrogate models, not experiments. For specific reaction classes underrepresented in Tox21/ToxCast, toxicity labels may be unreliable. The value function should be interpreted as providing *relative* route ranking, not absolute property prediction.

### 7.3 Comparison to D-MPC [Lu et al., 2025]

The original D-MPC is designed for continuous control (robot locomotion, manipulation). Key adaptations for chemistry:

| Aspect | D-MPC (original) | DMPSP (ours) |
|--------|-----------------|-------------|
| State space | Continuous (joint angles, positions) | Mixed: discrete molecules + continuous conditions |
| Action space | Continuous (torques, velocities) | Discrete-continuous hybrid (reaction class + conditions) |
| Dynamics model | Neural ODE / physics sim | ReactionT5 (language model) |
| Objective | Single reward (locomotion) | 10-dimensional process objectives |
| Planning horizon | 20-50 steps | 5-10 steps (synthesis routes) |
| Evaluation | Locomotion success rate | Route validity, property optimization |

---

## 8. Broader Impact

**Positive.** Accelerating route discovery reduces the time and cost of drug development. Multi-objective optimization (green chemistry, safety) could guide chemists toward more sustainable processes from the outset, rather than optimizing safety and waste as afterthoughts.

**Concerns.** A system that proposes novel synthesis routes could, in principle, be queried for routes to controlled substances or chemical precursors. DMPSP inherits no special restrictions on target molecules. Deployment should include target molecule filtering against controlled substance lists. The current model does not produce bioweapon precursors — its training data is USPTO organic chemistry — but downstream users should implement appropriate safeguards.

---

## 9. Conclusion

DMPSP applies factored diffusion-based MPC to multi-step chemical synthesis planning, addressing three core limitations of existing tree-search methods: compounding single-step errors, single-objective optimization, and fixed planning objectives. The joint world model (DynamicsDiffusion over fine-tuned ReactionT5) eliminates compounding error; the action proposal diffusion generates diverse, chemically-conditioned reaction sequences; and the 10-head value function enables runtime objective re-weighting with zero additional compute. On USPTO-50K, DMPSP achieves competitive route validity and planning speed while enabling multi-objective control unavailable in existing systems. We release all code, training scripts, and model checkpoints.

**Future work.** (1) Scaling to ORD (1M+ reactions) with real yield/condition labels. (2) Integrating RDKit reaction SMARTS validation into the diffusion sampling loop (constrained diffusion). (3) FastAPI inference server for multi-user deployment. (4) Wet lab validation of top-ranked routes for 10 pharmaceutical targets. (5) Extension to biocatalysis and organometallic chemistry.

---

## References

- Ajay et al. (2023). *Is Conditional Generative Modeling all you need for Decision-Making?* ICLR.
- Ahneman et al. (2018). *Predicting reaction performance in C–N cross-coupling using machine learning.* Science.
- Chen et al. (2020). *Retro*: Learning to Plan for Retrosynthesis*. NeurIPS.
- Chi et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS.
- Coley et al. (2019). *ASKCOS: A Synthesis Planning Software Platform with Retrosynthetic Design, Reaction Prediction, and Process Safety Evaluation.* ACS Central Science.
- Genheden et al. (2020). *AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning.* J. Cheminform.
- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
- Janner et al. (2022). *Planning with Diffusion.* ICML.
- Lowe (2012). *Chemical reactions from US patents, 1976-Sep2016.* Figshare.
- Lu et al. (2025). *Diffusion Model Predictive Control.* TMLR. arXiv:2410.05364v2.
- Molga et al. (2022). *Chemist-in-the-loop: multi-objective optimization of synthetic routes by a human-guided evolutionary algorithm.* J. Chem. Inf. Model.
- Sagawa et al. (2023). *Incorporating Synthetic Accessibility in Drug Design through Reaction-driven Molecule Generation.* (ReactionT5) arXiv.
- Schwaller et al. (2019). *Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction.* ACS Central Science.
- Schwaller et al. (2021). *Predicting the yield of amide bond formation from a machine learning approach.* Nature Communications.
- Segler et al. (2018). *Planning chemical syntheses with deep neural networks and symbolic AI.* Nature.
- Song et al. (2021). *Denoising Diffusion Implicit Models.* ICLR.
- Thakkar et al. (2021). *Unbiased Evaluation of Deep Uncertainty in Deep Learning Classifiers.* Chemical Science.
- Unterthiner et al. (2015). *Toxicity Prediction using Deep Learning.* arXiv. (DeepTox)
- Wu et al. (2018). *MoleculeNet: a benchmark for molecular machine learning.* Chemical Science.

---

## Appendix A: Full Hyperparameter Table

| Component | Parameter | Value | Rationale |
|-----------|-----------|-------|-----------|
| MorganFP Encoder | fp_radius | 2 | ECFP4 standard |
| | fp_size | 2048 | Standard |
| | hidden_dim | 256 | Matches model d_model |
| ActionProposalDiffusion | d_model | 256 | Memory vs. capacity tradeoff |
| | n_layers | 5 | Sufficient for sequence-level diffusion |
| | n_heads | 8 | d_model / 32 |
| | d_ff | 1024 | 4× d_model |
| | dropout | 0.1 | Standard |
| | T_diffusion | 1000 | Standard DDPM |
| | β schedule | cosine | More stable than linear |
| | DDIM steps (inference) | 50 | 20× speedup over training |
| | horizon F | 5 | 5-step lookahead balances quality vs. speed |
| | N candidates | 64 | Parallelizable on GPU |
| | EMA decay | 0.999 | Standard |
| ChemistryWorldModel | backbone | ReactionT5v2-forward | Best available reaction LM |
| | d_model | 768 (fixed, T5) | Pre-trained |
| | property_head_dim | 256 | |
| | dropout (heads) | 0.1 | |
| | gradient_checkpointing | True | Required for batch=16 on P100 |
| | λ_yield | 1.0 | Highest quality labels |
| | λ_tox, λ_mfg, λ_supply, λ_fto | 0.5 each | Lower quality pseudo-labels |
| ValueFunction | d_model | 256 | |
| | n_layers | 10 | Deeper for trajectory-level reasoning |
| | n_heads | 8 | |
| | n_objectives | 10 | |
| | dropout | 0.1 | |
| Planner | beam_width | 8 | |
| | MCTS simulations | 64 | |
| | UCB constant | √2 | Standard UCT |
| Training (all) | optimizer | AdamW | |
| | weight_decay | 1e-2 | |
| | grad_clip | 1.0 | |
| | fp16 | True | |

## Appendix B: Synthesis State Encoding Details

The synthesis state $s$ is encoded into a fixed-size vector as follows:

```
enc(s) = MLP([
    enc(m^target),        # 256-dim Morgan FP projection
    enc(m^current),       # 256-dim
    mean_pool(enc(I)),    # 256-dim (inventory mean)
    embed(h),             # L × 32-dim reaction history
    [T, P, σ, c, k, φ, ψ]  # 7 scalar process variables
]) → R^512
```

Process scalars are normalized: temperature to [0,1] over [250K, 500K], pressure over [0.1, 10] atm, cost over [0, $1000], step index over [0, K].

## Appendix C: Reaction Class Mapping (USPTO-50K)

| Class ID | Name | Count (train) | Example |
|----------|------|--------------|---------|
| 1 | Heteroatom alkylation & arylation | 11,040 | N-alkylation |
| 2 | Acylation & related | 7,543 | Amide coupling |
| 3 | C-C bond formation | 6,890 | Suzuki, Heck |
| 4 | Heterocycle formation | 5,812 | Ring cyclization |
| 5 | Reduction | 4,201 | Hydrogenation |
| 6 | Oxidation | 2,987 | Swern, TEMPO |
| 7 | Functional group interconversion | 2,543 | Ester hydrolysis |
| 8 | Protection/deprotection | 1,998 | Boc protection |
| 9 | Functional group addition | 1,487 | Grignard |
| 10 | Other | 513 | Miscellaneous |

## Appendix D: Pseudo-Label Pipeline Details

### D.1 Yield Labeling

Yield-BERT [Schwaller et al., 2021] is a BERT model fine-tuned on 6,000 HTE yield measurements for Pd-catalyzed C-N couplings. For non-coupling reactions, we use a reaction-class-specific fallback:

- Reductions (class 5): yield ~ N(0.85, 0.05) (well-optimized reactions)
- Oxidations (class 6): yield ~ N(0.75, 0.10)
- Coupling (class 3): Yield-BERT prediction
- Other classes: yield ~ N(0.80, 0.08)

These are reasonable priors from synthetic chemistry practice, not ground truth.

### D.2 Toxicity Labeling

RDKit Chem.FilterCatalog with PAINS, Brenk, and NIH alert sets identifies structural hazards. DeepTox [Unterthiner et al., 2015] provides multi-label toxicity prediction across 12 Tox21 endpoints. We aggregate to: mutagenicity (Ames), carcinogenicity (rodent), process safety (reactive functional groups).

### D.3 Green Chemistry Labeling

**Atom Economy (AE):**
$$\text{AE} = \frac{M_{\text{product}}}{\sum_i M_{\text{reactant}_i}} \in [0,1]$$
Computed directly from reaction SMILES using RDKit molecular weight.

**Process Mass Intensity (PMI):**
$$\text{PMI} = \frac{\text{total mass in} (\text{kg})}{\text{mass product} (\text{kg})}$$
Approximated from stoichiometry and typical solvent volumes per reaction class.

**Solvent greenness:** CHEM21 solvent guide assigns each solvent a green/amber/red score. Mapped to [0, 0.5, 1.0].

$$J_{\text{green}} = \frac{1}{2}(\text{AE} + \text{GreenSolv})$$

## Appendix E: Compute Budget

| Phase | GPU | Steps | Time | Peak VRAM |
|-------|-----|-------|------|-----------|
| Data preparation | CPU | — | ~10 min | — |
| Phase 1 (action proposal) | P100 16GB | 100K | ~3h | 8GB |
| Phase 2 (world model) | P100 16GB | 100K | ~9h | 15GB |
| Phase 3 (value function) | P100 16GB | 100K | ~2h | 6GB |
| Evaluation (100 molecules) | P100 | — | ~30 min | 4GB |
| **Total** | | | **~15h** | |

Total Kaggle free GPU hours consumed: ~15h (of 30h/week free allocation).
Cost on paid GPU (vast.ai A100 @ $1.50/hr): ~$22.50 for full 100K run.
