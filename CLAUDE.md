# CLAUDE.md — DMPSP

## What

Diffusion Model Predictive Synthesis Planning — controllable synthesis world model for pharma route discovery.

NOT a retrosynthesis tree search. D-MPC's factored diffusion applied to chemistry.
Open source. Clean. Nails the research.

Based on: arXiv 2410.05364v2 (D-MPC, TMLR May 2025)

## Technical Novelty

1. First D-MPC factored diffusion (action proposal × dynamics) applied to synthesis planning
2. Joint multi-step world model — avoids compounding prediction errors across 10-step routes
3. Runtime 10-objective MPC — no retraining when objective weights change
4. Process-aware state: inventory + conditions + IP proximity (not just molecules)
5. ReactionT5 fine-tuned as world model backbone — leverages 10B+ pre-training

## Architecture (4 trainable components)

| Component | Role | Architecture |
|-----------|------|-------------|
| `ActionProposalDiffusion` ρ | Proposes N×F reaction sequences | DDPM, 5-layer Transformer, cross-attn cond |
| `ChemistryWorldModel` p_d | Predicts F-step reaction outcomes | ReactionT5 fine-tuned + 5 property heads + DDIM wrapper |
| `ValueFunction` J | Scores trajectories (10 objectives) | 10-layer Transformer, 10-head regression |
| `DMPSPPlanner` | MPC loop | Beam search (default) + MCTS |

## SynthesisState (rich, not just molecule)

```python
target_smiles | current_smiles | inventory | reaction_history |
temperature | pressure | scale | cost_accumulated | step_number |
yield_so_far | purity_so_far
```

Encoder: GINEncoder (default, PyG) or ChemBERTaEncoder (HuggingFace, optional). Config-driven.

## World Model: ReactionT5 + 5 property heads

Pre-trained backbone: `sagawa/ReactionT5` (HuggingFace, Apache 2.0)
Fine-tune on task data. Add:
- YieldHead
- ToxicityHead (mutagenicity, carcinogenicity, process safety)
- ManufacturabilityHead (scale-up feasibility, GMP)
- SupplyChainHead (availability, lead times)
- PatentabilityHead (route novelty, FTO risk)

Wrapped in DynamicsDiffusion (DDIM, joint F-step, non-causal).

## 10 Objectives (ValueFunction, runtime weights)

yield | purity | cost | novelty | fto_risk | green_chem | manufacturability | safety | robustness | supply_avail

Different users tune weights — no retraining:
- Generic pharma: cost + fto_risk
- CDMO: manufacturability + robustness
- Discovery: novelty + yield
- Green: green_chem + safety

## Dataset Strategy

**Priority 1**: User CSV (patent scrape / high-quality sources). Column names from `configs/data.yaml`.
**The moat**: Process outcome data — failures, impurity profiles, scale-up failures, safety incidents.
**Fallbacks**: USPTO-50K, ORD, ChEMBL (auto-downloaded).

## GPU Plan

- Kaggle P100 (30h/week free): world model fine-tuning, proposal model
- Lightning.ai A10G (22h/month free): experimentation
- vast.ai A100 (~$1.50/hr): full training run (~12h = ~$18)

## Stack

- Python 3.11+, PyTorch 2.4+, PyTorch Geometric 2.4+
- RDKit 2024, Transformers (HuggingFace), FastAPI, httpx, argparse
- ReactionT5 backbone (`sagawa/ReactionT5`)
- ADMETlab 3.0 API, eMolecules API

## Package Structure

Package at root: `dmpsp/` — imports as `from dmpsp import DMPSPPlanner`

15 Python source files total:
```
dmpsp/
├── state.py            # SynthesisState, SynthesisAction, SynthesisRoute
├── encoder.py          # MolecularEncoder (GIN | ChemBERTa)
├── diffusion.py        # DDPM/DDIM utils
├── action_proposal.py  # ActionProposalDiffusion
├── world_model.py      # ChemistryWorldModel (ReactionT5 + heads + DynamicsDiffusion)
├── value_fn.py         # ValueFunction (10-head)
├── scorer.py           # ExternalScorer (ADMETlab + eMolecules, real APIs)
├── planner.py          # DMPSPPlanner (beam + MCTS)
└── utils.py            # chemistry helpers, checkpointing, logging
data/
├── loader.py
├── dataset.py
└── preprocess.py
train/
├── train_world_model.py
├── train_proposal.py
└── train_value.py
scripts/
├── prepare_data.py
└── plan_route.py
notebooks/
├── 01_train_kaggle.ipynb
└── 02_demo.ipynb
```

## Rules (from Proejct rules.md)

- No hardcoding. No faking. Raise on failure.
- argparse for scripts. `logging.getLogger(__name__)` in library.
- Tests mirror source. Lazy data loading.
- requirements.txt exact pins.

## Repo State

All phases complete. 67/68 tests passing (1 skipped: ADMETlab live API, no key set).

### What's built
- `dmpsp/` — full package: state, encoder, diffusion utils, action_proposal, world_model, value_fn, scorer, planner, utils
- `data/` — loader (CSV/USPTO/ORD/ChEMBL), preprocess, dataset
- `train/` — train_proposal, train_world_model, train_value
- `scripts/` — prepare_data.py, plan_route.py
- `tests/` — 68 tests covering all modules
- `configs/` — model.yaml, data.yaml, train.yaml
- `README.md`, `LICENSE`, `pyproject.toml`, `requirements.txt`

### Next steps
1. Prepare data: `python scripts/prepare_data.py --source csv --data_path YOUR_CSV.csv --out_dir data/processed/`
2. Train on Kaggle/Lightning.ai/vast.ai using `train/` scripts
3. Run inference: `python scripts/plan_route.py --smiles SMILES --checkpoint_dir checkpoints/`
