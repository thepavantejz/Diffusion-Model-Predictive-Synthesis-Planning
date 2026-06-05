# DMPSP: Diffusion Model Predictive Synthesis Planning

A **controllable synthesis planning world model** for pharmaceutical route discovery.

Given a target drug molecule (SMILES), DMPSP finds optimal multi-step synthetic routes by combining:
- **Diffusion-based action proposal** — proposes diverse reaction sequences
- **Chemistry world model** (ReactionT5 fine-tuned) — predicts multi-step reaction outcomes
- **Multi-objective MPC** — plans routes optimized for 10 simultaneous objectives

Based on **D-MPC** (arXiv:2410.05364, TMLR 2025), adapted for pharmaceutical synthesis.

---

## Why not just another retrosynthesis tool?

Existing tools (ASKCOS, AiZynthFinder) predict the *next single reaction step*. DMPSP plans *entire synthesis trajectories* using a learned world model, similar to how AlphaGo imagines future game states. This avoids the compounding errors of greedy step-by-step planning.

Key advantages:
- **Runtime multi-objective tuning**: change weights without retraining (yield vs. cost vs. safety vs. IP)
- **Process-aware**: considers manufacturing constraints, supply chain, green chemistry
- **Patent-aware**: flags IP risk at route planning time
- **Adaptable**: fine-tune only the dynamics model when new catalysts/conditions become available

---

## Architecture

```
Target SMILES
    │
    ▼
SynthesisState (molecule + inventory + conditions + cost + IP proximity)
    │
    ▼ ρ — ActionProposalDiffusion
N candidate reaction sequences (DDPM, 32 steps)
    │
    ▼ p_d — ChemistryWorldModel (ReactionT5 + property heads, DDIM 10 steps)
N predicted synthesis trajectories (joint F-step, non-causal)
    │
    ▼ J — ValueFunction (10-head regression, runtime weights)
Ranked candidates
    │
    ▼ DMPSPPlanner (beam search / MCTS)
Best synthesis route + all 10 objective scores
```

**10 objectives** (runtime-tunable, no retraining):
`yield` · `purity` · `cost` · `novelty` · `fto_risk` · `green_chem` · `manufacturability` · `safety` · `robustness` · `supply_avail`

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/dmpsp.git
cd dmpsp
pip install -r requirements.txt
pip install -e .
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add:
#   ADMETLAB_API_KEY  — from https://admetlab3.scbdd.com (free)
#   EMOLECULES_API_KEY — from https://www.emolecules.com
#   HF_TOKEN          — from https://huggingface.co/settings/tokens
```

### 3. Prepare data

```bash
# Option A: your own reaction data (recommended — higher quality)
python scripts/prepare_data.py \
    --source csv \
    --data_path data/raw/your_reactions.csv \
    --data_config configs/data.yaml \
    --out_dir data/processed/

# Option B: public USPTO-50K
python scripts/prepare_data.py \
    --source uspto50k \
    --out_dir data/processed/
```

### 4. Train

```bash
# Train world model (ReactionT5 fine-tuning)
python train/train_world_model.py \
    --model_config configs/model.yaml \
    --train_config configs/train.yaml \
    --data_dir data/processed/ \
    --out_dir checkpoints/world_model/ \
    --device cuda

# Train action proposal
python train/train_proposal.py \
    --model_config configs/model.yaml \
    --train_config configs/train.yaml \
    --data_dir data/processed/ \
    --out_dir checkpoints/action_proposal/ \
    --device cuda

# Train value function
python train/train_value.py \
    --model_config configs/model.yaml \
    --train_config configs/train.yaml \
    --data_dir data/processed/ \
    --out_dir checkpoints/value_fn/ \
    --device cuda
```

### 5. Find synthesis routes

```bash
python scripts/plan_route.py \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --weights_json '{"yield":0.3,"cost":0.2,"safety":0.2,"manufacturability":0.15,"fto_risk":0.15}' \
    --checkpoint_dir checkpoints/ \
    --max_steps 5
```

---

## Training on free GPUs

### Kaggle (30h/week free, P100 16GB)

Open `notebooks/01_train_kaggle.ipynb` on Kaggle. Upload your processed data as a Kaggle dataset. Gradient checkpointing is enabled by default for 16GB compatibility.

### Lightning.ai (22h/month free, A10G 24GB)

Clone the repo in a Lightning Studio. Run training scripts directly — no notebook required.

### vast.ai (A100 80GB, ~$1.50/hr)

For full training runs (~10-12 hours total across all 3 models):
```bash
# On a vast.ai A100 instance:
pip install -r requirements.txt && pip install -e .
python train/train_world_model.py --device cuda ...
python train/train_proposal.py --device cuda ...
python train/train_value.py --device cuda ...
```

---

## Data format

Your CSV file should have these columns (names configurable in `configs/data.yaml`):

| Column | Required | Description |
|--------|----------|-------------|
| `reactant_smiles` | ✓ | SMILES (comma-separated for multiple reactants) |
| `product_smiles` | ✓ | Main product SMILES |
| `reaction_class_id` | ✓ | Integer reaction class (0–99) |
| `temperature` | optional | Temperature in Kelvin |
| `yield_percent` | optional | Yield in % [0–100] |
| `purity_percent` | optional | Purity in % [0–100] |
| `cost_usd` | optional | Starting material cost in USD/g |
| `solvent` | optional | Solvent SMILES or name |
| `catalyst` | optional | Catalyst SMILES or name |

**What data gives the best models**: Include failed reactions, impurity profiles, scale-up failures, and safety incidents — not just successful reactions. This process outcome data is what makes the world model genuinely useful.

---

## Run tests

```bash
pytest tests/ -v --tb=short
```

Tests that require API keys (ADMETlab, eMolecules) are automatically skipped if keys are not set.

---

## Project structure

```
dmpsp/          — Python package (model code)
data/           — data loading and preprocessing
train/          — training scripts
scripts/        — CLI tools (prepare_data.py, plan_route.py)
notebooks/      — Kaggle training notebook, demo notebook
tests/          — test suite (mirrors dmpsp/ structure)
configs/        — YAML configs (model, data, training)
```

---

## Citation

If you use DMPSP, please cite the D-MPC paper it is based on:

```bibtex
@article{dmpc2024,
  title={Diffusion Model Predictive Control},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://arxiv.org/abs/2410.05364}
}
```

---

## License

MIT
