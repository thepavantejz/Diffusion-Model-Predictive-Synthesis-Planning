# DMPSP — Production Roadmap

## Current State (as of 2026-06-06)

- All 3 training phases running on Kaggle P100 (100K steps each)
- Data: USPTO-50K (50,016 reactions) — real SMILES, synthetic property labels
- Property heads train on imputed/zero labels → directionally wrong signals
- CLI inference working (`scripts/plan_route.py`)
- No serving layer, no validation, no real property scores

---

## Gap 1: Data — The Real Bottleneck

### Problem
All 5 property heads (yield, toxicity, manufacturability, supply chain, patentability)
currently train on synthetic/imputed labels. Real proprietary process data
(yields, conditions, failures, scale-up records) is inaccessible IP.

### Solution: Pseudo-Label Pipeline

Build `scripts/generate_labels.py` — surrogate models + computed properties +
public APIs. No proprietary data needed.

| Objective | Method | Source |
|-----------|--------|--------|
| `yield` | Yield-BERT inference | ORD (1M reactions w/ real yields) |
| `toxicity` | QSAR model (DeepTox) | Tox21, ToxCast, ClinTox (public) |
| `manufacturability` | PMI + temp extremes + solvent score | CHEM21 solvent guide (computable) |
| `supply_avail` | Price/availability lookup | eMolecules free API, Sigma-Aldrich |
| `fto_risk` | Patent coverage check | SureChEMBL public API |
| `green_chem` | Atom economy + GSK solvent score | RDKit (computable) |
| `safety` | Structural alerts + COSHH | RDKit alerts + SDS data |
| `novelty` | Tanimoto vs known routes | USPTO-50K itself |
| `cost` | Starting material price + step count | eMolecules API |
| `purity` | Side reaction likelihood | Reaction class + conditions heuristics |

**Files to build:**
```
scripts/
├── generate_labels.py      — orchestrator
├── labelers/
│   ├── yield_labeler.py    — Yield-BERT on reaction SMILES
│   ├── tox_labeler.py      — DeepTox / RDKit structural alerts
│   ├── green_labeler.py    — atom economy, PMI, solvent score
│   ├── supply_labeler.py   — eMolecules API per starting material
│   ├── patent_labeler.py   — SureChEMBL API per route
│   └── merge_labels.py     — join → augmented trajectories_train.pkl
```

**Quality note:** Pseudo-labels are bounded by surrogate model quality.
Yield-BERT on Suzuki: R²≈0.85. Toxicity QSAR: decent. Manufacturability: rough proxy.
Even noisy labels beat random — value function learns relative route ranking,
not exact numbers.

**Path to real data:**
- ORD (Open Reaction Database): 1M+ reactions, real conditions and some yields. Free.
- If partnered with CRO/pharma: their yield CSV is the actual moat.

---

## Gap 2: External APIs Not Wired

- `scorer.py`: ADMETlab (toxicity) and eMolecules (supply chain) are stubbed
- Need real API keys for live scores during inference
- ADMETlab 3.0 has a free tier (rate-limited)
- eMolecules has a free API tier

**To do:** Wire real API calls in `scorer.py`, add env var config for keys.

---

## Gap 3: No Chemical Validity Guarantee

- World model outputs product SMILES — no guarantee chemically valid
- No check that proposed reaction is feasible given reagents
- **Fix:** RDKit validity filter + reaction SMARTS check on each planner step

---

## Gap 4: No Serving Layer

- Only CLI today (`plan_route.py`)
- Stack already includes FastAPI
- **To build:**
  - `api/server.py` — FastAPI app
  - POST `/plan` — takes SMILES + weights, returns route JSON
  - GET `/health`
  - Auth (API key)
  - Async inference (background tasks)
  - Docker image

---

## Gap 5: No Baseline Comparison

- Not benchmarked vs AiZynthFinder, ASKCOS, Retro*
- Need ablation: beam vs MCTS, with/without value fn, 1-obj vs 10-obj
- **To do:** `scripts/benchmark.py` already exists — run it, publish numbers

---

## Gap 6: No Wet Lab Validation

- Zero experimental confirmation routes actually work
- Model quality bounded by training distribution
- Long-term: partner with a CRO or academic lab to validate top-K routes

---

## Training Steps

| Quality | Steps/phase | Total GPU | Cost |
|---------|-------------|-----------|------|
| Demo/research (current) | 100K | ~15h | Free Kaggle |
| Robust demo | 300K | ~45h | ~3 Kaggle sessions |
| Production | 1M+ | ~150h | ~$225 vast.ai A100 |

Steps are not the bottleneck — data quality is.

---

## Priority Order (what to do next)

1. **Finish current 100K training** — phases 2 + 3 still running on Kaggle
2. **Download checkpoints** — zip from Kaggle Output tab
3. **Run smoke test + benchmark** — Cell 9 + Cell 11 on Kaggle
4. **Build pseudo-label pipeline** — `scripts/generate_labels.py` + labelers
5. **Retrain with real labels** — 300K steps on labeled data
6. **Wire ADMETlab + eMolecules APIs** — real scores in scorer.py
7. **Add validity filter** — RDKit check on each planner step
8. **Build FastAPI server** — serve inference via REST
9. **Benchmark vs AiZynthFinder** — publish comparison
10. **Wet lab validation** — find a collaborator
