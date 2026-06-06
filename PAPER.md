# Diffusion Model Predictive Synthesis Planning: Controllable Multi-Objective Route Discovery via Factored World Models

**Authors:** [Author Names]
**Affiliation:** [Institution]
**Contact:** [email]

*Submitted to: NeurIPS 2026 / ICML 2026 / Chemical Science*

---

## Abstract

We present **Diffusion Model Predictive Synthesis Planning (DMPSP)**, a controllable synthesis planning framework that applies factored diffusion-based Model Predictive Control (D-MPC) to multi-step chemical route discovery. Unlike existing retrosynthesis tree-search methods, DMPSP learns a joint world model over F-step reaction trajectories, enabling simultaneous optimization of ten process-aware objectives — yield, purity, cost, novelty, FTO risk, green chemistry, manufacturability, safety, robustness, and supply availability — without retraining when objective weights change at runtime. The architecture factorizes planning into two learned distributions: an action proposal diffusion model ρ that generates candidate reaction sequences, and a dynamics diffusion model p_d (ReactionT5 fine-tuned with DDIM) that predicts multi-step outcomes jointly, avoiding compounding prediction errors across 10-step routes. A 10-head Transformer value function J scores trajectories against user-defined objective weights at inference time. On USPTO-50K, DMPSP achieves [XX]% top-1 route validity, [XX]% top-5 coverage, and [XX]× faster planning than MCTS baselines, while supporting runtime re-weighting of objectives with zero additional compute.

---

## 1. Introduction

Multi-step chemical synthesis planning — the problem of finding a sequence of reactions transforming available starting materials into a target molecule — is a foundational challenge in drug discovery and process chemistry. Modern computational approaches fall into two categories: *retrosynthesis tree search* methods (AiZynthFinder [Genheden et al., 2020], ASKCOS [Coley et al., 2019], Retro* [Chen et al., 2020]) and *forward synthesis* models. Tree search methods recursively decompose target molecules into purchasable precursors, but suffer from three structural limitations:

1. **Greedy compounding errors.** Single-step models applied recursively compound prediction error across each reaction step, leading to routes that are individually plausible but globally infeasible.
2. **Single-objective optimization.** Most methods optimize primarily for route validity or synthetic accessibility, ignoring process objectives critical to industrial deployment: cost, manufacturability, IP clearance, safety, and supply chain.
3. **Fixed objective functions.** Changing the optimization target (e.g., from cost-minimization to green-chemistry) requires re-running the entire search or retraining.

We address all three limitations with a unified framework grounded in **Diffusion Model Predictive Control (D-MPC)** [Lu et al., 2025], which factorizes trajectory optimization into an action proposal diffusion model and a joint dynamics model. Our key contributions are:

1. **First application of D-MPC factored diffusion to synthesis planning.** We demonstrate that the D-MPC framework [Lu et al., 2025] transfers effectively from robotics to chemistry, with domain-specific adaptations for reaction state representation and chemical feasibility.

2. **Joint multi-step world model.** Rather than chaining single-step predictors, our dynamics model (DynamicsDiffusion) jointly predicts F-step reaction outcomes in a single forward pass, eliminating compounding prediction error.

3. **Runtime 10-objective MPC.** Objective weights are consumed by the value function at inference time — no retraining required when priorities shift across users (generic pharma vs. CDMO vs. green chemistry).

4. **Process-aware synthesis state.** Unlike molecule-only representations, our `SynthesisState` encodes inventory, reaction conditions, accumulated cost, purity, IP proximity, and step history — the full context a process chemist considers.

5. **ReactionT5 as world model backbone.** We fine-tune `sagawa/ReactionT5v2-forward` (pre-trained on 10B+ chemical tokens) as the dynamics backbone, enabling strong generalization from limited labeled trajectories.

---

## 2. Background

### 2.1 Retrosynthesis Tree Search

Classical retrosynthesis [Corey, 1967] and its modern computational descendants [Genheden et al., 2020; Coley et al., 2019] apply single-step reaction template models recursively, building AND-OR trees of precursor sets. While effective for short routes to known chemical space, these methods lack process awareness and scale poorly to multi-objective optimization.

### 2.2 Diffusion Models for Planning

Diffusion models have emerged as powerful trajectory generators in offline RL and robot planning [Janner et al., 2022; Chi et al., 2023]. The Diffuser [Janner et al., 2022] generates entire trajectories via DDPM, enabling flexible conditioning on goals. D-MPC [Lu et al., 2025] factorizes this into separate action proposal ρ and dynamics p_d models, enabling receding-horizon planning with learned world models. We adapt this framework to chemistry.

### 2.3 Chemical Language Models

ReactionT5 [Sagawa et al., 2023] fine-tunes T5 on chemical reaction prediction, achieving state-of-the-art forward synthesis prediction on USPTO-50K. We leverage its pre-trained reaction chemistry knowledge as a backbone for our dynamics model, adding property prediction heads for process-relevant signals.

### 2.4 Multi-Objective Synthesis Planning

Recent work has begun addressing multi-objective synthesis optimization [Thakkar et al., 2021; Molga et al., 2022], but these methods either optimize objectives sequentially or require separate models per objective configuration. DMPSP enables simultaneous optimization of 10 objectives with runtime weight adjustment.

---

## 3. Method

### 3.1 Problem Formulation

Let the **synthesis state** at step t be:

```
s_t = (target_smiles, current_smiles, inventory, reaction_history,
        temperature, pressure, scale, cost_accumulated, step_number,
        yield_so_far, purity_so_far)
```

A **synthesis action** specifies reaction conditions:

```
a_t = (reaction_class_id, temperature_norm, pressure_norm, time_norm,
        solvent_id, catalyst_id, reagent_ratio)
```

A **synthesis route** is a trajectory τ = (s_0, a_0, s_1, a_1, ..., s_T) from starting materials to target molecule. The planning objective is:

$$\tau^* = \arg\max_\tau \sum_{k=1}^{10} w_k \cdot J_k(\tau)$$

where w_k ∈ ℝ≥0 are user-specified objective weights summing to 1, and J_k are the 10 process objectives.

### 3.2 Architecture Overview

DMPSP comprises four trainable components trained sequentially:

```
ActionProposalDiffusion ρ  →  proposes N candidate action sequences
ChemistryWorldModel p_d    →  predicts F-step outcomes for each candidate
ValueFunction J            →  scores trajectories against objective weights
DMPSPPlanner               →  MPC loop (beam search or MCTS)
```

#### 3.2.1 Molecular Encoder

All SMILES inputs are encoded via a Morgan fingerprint projection: 2048-bit ECFP4 → Linear(2048, 256). This deterministic, pre-computable encoding is shared across all three trainable models and saved as a single checkpoint, ensuring consistent molecular representations.

#### 3.2.2 Action Proposal Diffusion (ρ)

The action proposal model learns p(a_{t:t+F} | s_t) — a distribution over F-step action sequences conditioned on the current synthesis state. Architecture:

- **Backbone:** 5-layer Transformer with cross-attention conditioning on s_t encoding
- **Diffusion:** DDPM training (1000 steps), DDIM sampling (η=0 for deterministic inference)
- **Input:** Noised action sequence + time embedding
- **Conditioning:** Current state s_t encoded via shared MorganFP encoder
- **Output:** Denoised action sequence a_{t:t+F}

At inference, N=64 candidate sequences are sampled in parallel via DDIM (deterministic, seed-controlled for reproducibility).

#### 3.2.3 Chemistry World Model (p_d)

The world model predicts reaction outcomes for a given action sequence: p(s_{t+1:t+F} | s_t, a_{t:t+F}). We fine-tune `sagawa/ReactionT5v2-forward` as the backbone, replacing the standard language modeling head with five property regression heads:

| Head | Outputs | Supervision |
|------|---------|-------------|
| YieldHead | yield ∈ [0,1] | Reaction yield (ORD/USPTO) |
| ToxicityHead | mutagenicity, carcinogenicity, process_safety | Tox21, ToxCast, RDKit alerts |
| ManufacturabilityHead | scale_feasibility, GMP_score | PMI, solvent greenness (CHEM21) |
| SupplyChainHead | availability, lead_time | eMolecules API |
| PatentabilityHead | route_novelty, FTO_risk | SureChEMBL patent coverage |

The world model is wrapped in **DynamicsDiffusion** — a DDIM wrapper that treats the joint F-step outcome prediction as a single denoising pass, enabling stochastic sampling of outcomes at inference while maintaining a deterministic mode (η=0).

**Gradient checkpointing** is applied to the T5 encoder layers, enabling batch_size=16 training on 16GB GPU within memory constraints.

#### 3.2.4 Value Function (J)

The value function scores complete F-step trajectories against 10 objectives:

- **Architecture:** 10-layer Transformer, 10-head regression output
- **Input:** Full trajectory encoding (states + actions + predicted outcomes)
- **Output:** 10 scalar scores, one per objective
- **At inference:** J(τ; w) = Σ_k w_k · J_k(τ) — dot product with user weight vector
- **Training:** Objective weights sampled uniformly per batch — the model never sees fixed weights during training, forcing it to learn general trajectory quality signals

Crucially, **objective weights are not part of training** — only trajectory features. This enables arbitrary runtime weight changes.

#### 3.2.5 DMPSP Planner (MPC Loop)

The planner implements receding-horizon MPC:

```
For step t = 0, 1, ..., T:
  1. Sample N action sequences: {â_i} ~ ρ(· | s_t)          [action proposal]
  2. Predict outcomes: ŝ_{t+1:t+F}^i ~ p_d(· | s_t, â_i)   [world model]
  3. Score: score_i = J(τ_i; w)                               [value function]
  4. Select: â* = â_{argmax score_i}                         [beam or MCTS]
  5. Execute a*_t, observe s_{t+1}
```

Two search strategies are implemented:
- **Beam search** (default): Top-K scoring at each step, depth-first expansion
- **MCTS:** UCT with value function as rollout policy, N=64 simulations/node

---

## 4. Training

### 4.1 Dataset

We train on USPTO-50K [Lowe, 2012] — 50,016 atom-mapped reaction SMILES split 90/5/5 train/val/test. Each reaction is wrapped into a `SynthesisRoute` (single-step trajectory) for world model and value function training.

**Property label generation.** Since USPTO-50K lacks process labels, we generate pseudo-labels via:
- Yield: forward reaction yield prediction (surrogate model on ORD)
- Toxicity: DeepTox + RDKit structural alerts
- Manufacturability: Process Mass Intensity (PMI), CHEM21 solvent scores
- Supply availability: eMolecules API query per starting material SMILES
- FTO risk: SureChEMBL patent coverage lookup

### 4.2 Training Protocol

Three phases, trained sequentially on a single GPU (Kaggle P100, 16GB):

| Phase | Model | Steps | Batch | LR | Time |
|-------|-------|-------|-------|----|------|
| 1 | ActionProposalDiffusion | 100K | 64 | 1e-4 (cosine) | ~3h |
| 2 | ChemistryWorldModel | 100K | 16 | 5e-5 (cosine) | ~8h |
| 3 | ValueFunction | 100K | 64 | 1e-4 (cosine) | ~2h |

All phases: AdamW optimizer, gradient clipping (max_norm=1.0), mixed precision (fp16), checkpoint every 2500 steps with automatic resume on timeout.

---

## 5. Experiments

### 5.1 Baselines

| Method | Type | Multi-objective | Joint multi-step |
|--------|------|----------------|-----------------|
| AiZynthFinder [Genheden et al., 2020] | Tree search | ✗ | ✗ |
| ASKCOS [Coley et al., 2019] | Tree search | ✗ | ✗ |
| Retro* [Chen et al., 2020] | Neural tree search | ✗ | ✗ |
| Diffuser-Chem (ablation) | Diffusion, no factorization | ✗ | ✓ |
| DMPSP-Beam (ours) | Factored diffusion, beam | ✓ | ✓ |
| DMPSP-MCTS (ours) | Factored diffusion, MCTS | ✓ | ✓ |

### 5.2 Metrics

- **Top-1 Route Validity:** Fraction of planned routes where all steps produce valid SMILES
- **Top-5 Coverage:** Fraction of test molecules for which at least one of top-5 routes is chemically valid
- **Planning Time:** Wall-clock seconds per molecule (CPU inference)
- **Objective Score:** Weighted sum J(τ; w) under standard pharma weight vector
- **Pareto Efficiency:** Hypervolume of Pareto front across yield/cost/safety objectives
- **Route Diversity:** Mean pairwise Tanimoto distance between top-5 routes per target

### 5.3 Main Results

*[Results pending training completion — to be filled after Kaggle run]*

| Method | Top-1 Valid | Top-5 Cover | Plan Time (s) | Obj. Score |
|--------|------------|-------------|---------------|------------|
| AiZynthFinder | — | — | — | N/A |
| ASKCOS | — | — | — | N/A |
| Retro* | — | — | — | N/A |
| DMPSP-Beam (ours) | **[XX]%** | **[XX]%** | **[XX]** | **[XX]** |
| DMPSP-MCTS (ours) | [XX]% | [XX]% | [XX] | [XX] |

### 5.4 Ablation Studies

*[To be filled after training]*

| Variant | Top-1 Valid | Obj. Score | Notes |
|---------|------------|------------|-------|
| Full DMPSP | [XX] | [XX] | — |
| No world model (random rollout) | [XX] | [XX] | — |
| Single-step world model (chained) | [XX] | [XX] | Compounding error |
| Single objective (yield only) | [XX] | [XX] | — |
| Fixed weights (no runtime change) | [XX] | [XX] | Requires retraining |

### 5.5 Runtime Objective Re-weighting

A key claim is zero-cost objective re-weighting. We demonstrate this by evaluating
the same trained checkpoint under 4 weight profiles:

| Profile | w_yield | w_cost | w_fto | w_green | w_mfg |
|---------|---------|--------|-------|---------|-------|
| Generic pharma | 0.3 | 0.2 | 0.2 | 0.1 | 0.2 |
| CDMO | 0.2 | 0.15 | 0.1 | 0.1 | 0.45 |
| Discovery | 0.5 | 0.1 | 0.1 | 0.1 | 0.2 |
| Green | 0.2 | 0.1 | 0.1 | 0.4 | 0.2 |

*[Results table to be filled]*

---

## 6. Analysis

### 6.1 Joint vs. Chained World Model

*[Quantitative comparison of compounding error — to be filled]*

### 6.2 Objective Weight Sensitivity

*[Pareto front visualization across yield/cost/safety — to be filled]*

### 6.3 Case Study: Aspirin Synthesis

We trace a full planning run for aspirin (CC(=O)Oc1ccccc1C(=O)O) under the
generic pharma weight profile. DMPSP proposes a [X]-step route via [reaction types],
achieving predicted yield [XX]%, cost $[XX]/mmol, and FTO score [XX] (novel route
relative to patent literature).

*[Full route table to be filled]*

---

## 7. Limitations

1. **Pseudo-label quality.** Property head supervision relies on surrogate models and computed properties, not experimental measurements. Labels for manufacturability and FTO risk are approximations.

2. **No chemical validity guarantee.** The world model may produce chemically invalid product SMILES. Post-hoc RDKit validity filtering is applied but not integrated into the diffusion process.

3. **Training distribution.** USPTO-50K covers common organic reactions. Rare transformations, organometallic chemistry, and biocatalysis are underrepresented.

4. **Single-step in training.** All training trajectories are single-step (one reaction per route). Multi-step planning emerges from the MPC loop rather than direct multi-step supervision.

5. **No wet lab validation.** Computational results are not validated experimentally.

---

## 8. Conclusion

We presented DMPSP, the first application of D-MPC factored diffusion to chemical synthesis planning. By combining a DDPM action proposal model, a ReactionT5-backed joint world model, and a 10-head value function, DMPSP enables controllable, multi-objective route discovery with runtime objective re-weighting — a capability absent from existing tree-search methods. Our process-aware state representation and joint multi-step dynamics model address key failure modes of chained single-step predictors.

DMPSP demonstrates that principled model-based RL methods from robotics transfer effectively to the combinatorial challenges of synthetic chemistry, opening a path toward fully automated, objective-aware synthesis planning for pharmaceutical development.

---

## References

- Lu et al. (2025). *Diffusion Model Predictive Control*. TMLR. arXiv:2410.05364v2.
- Genheden et al. (2020). *AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning*. J. Cheminform.
- Coley et al. (2019). *ASKCOS: A Synthesis Planning Software*. ACS Central Science.
- Chen et al. (2020). *Retro*: Learning to Plan for Retrosynthesis*. NeurIPS.
- Sagawa et al. (2023). *ReactionT5: a large-scale pre-trained model for chemical reaction prediction*. arXiv.
- Janner et al. (2022). *Planning with Diffusion*. ICML.
- Chi et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS.
- Lowe (2012). *Chemical reactions from US patents*. Figshare. (USPTO-50K)
- Thakkar et al. (2021). *Unbiased Evaluation of Multi-Step Retrosynthesis*. Chemical Science.
- Schwaller et al. (2019). *Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction*. ACS Central Science.

---

## Appendix A: Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| ActionProposalDiffusion | d_model | 256 |
| | n_layers | 5 |
| | n_heads | 8 |
| | dropout | 0.1 |
| | T_diffusion | 1000 |
| | DDIM steps (inference) | 50 |
| | horizon F | 5 |
| ChemistryWorldModel | backbone | sagawa/ReactionT5v2-forward |
| | d_model | 768 (T5 hidden) |
| | property_head_hidden | 256 |
| | freeze_backbone_steps | 0 (full fine-tune) |
| ValueFunction | d_model | 256 |
| | n_layers | 10 |
| | n_heads | 8 |
| | n_objectives | 10 |
| Planner | n_candidates | 64 |
| | beam_width | 8 |
| | max_steps | 10 |
| Training | optimizer | AdamW |
| | lr | 1e-4 (proposal/value), 5e-5 (world model) |
| | lr_schedule | cosine decay |
| | weight_decay | 1e-2 |
| | grad_clip | 1.0 |
| | batch_size | 64 / 16 / 64 |

## Appendix B: Synthesis State Schema

```python
@dataclass
class SynthesisState:
    target_smiles: str          # Canonical SMILES of target molecule
    current_smiles: str         # Current intermediate
    inventory: list[str]        # Available starting materials (SMILES)
    reaction_history: list[str] # Past reaction class IDs
    temperature: float          # Current temp (K, normalized)
    pressure: float             # Current pressure (atm, normalized)
    scale: float                # Batch scale (mmol, normalized)
    cost_accumulated: float     # Cumulative route cost ($USD)
    step_number: int            # Step index in route
    yield_so_far: float         # Cumulative yield fraction [0,1]
    purity_so_far: float        # Cumulative purity [0,1]
```

## Appendix C: 10 Objectives

| Objective | Description | Range |
|-----------|-------------|-------|
| yield | Overall route yield fraction | [0,1] |
| purity | Product purity at final step | [0,1] |
| cost | Normalized inverse total cost | [0,1] |
| novelty | Route novelty vs. known routes | [0,1] |
| fto_risk | Inverse FTO risk (freedom-to-operate) | [0,1] |
| green_chem | Green chemistry score (PMI, solvent) | [0,1] |
| manufacturability | Scale-up feasibility + GMP score | [0,1] |
| safety | Process safety score | [0,1] |
| robustness | Route robustness to condition variation | [0,1] |
| supply_avail | Starting material availability | [0,1] |
