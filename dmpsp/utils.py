"""Shared utilities: chemistry helpers, checkpointing, logging setup.

All functions here have no dependencies on other dmpsp modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

logger = logging.getLogger(__name__)

# Atom types used for node featurization. 'other' catches any element not listed.
_ATOM_TYPES: list[str] = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
    "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "other",
]
_ATOM_TYPE_INDEX: dict[str, int] = {a: i for i, a in enumerate(_ATOM_TYPES)}

NODE_FEAT_DIM: int = 9
EDGE_FEAT_DIM: int = 5


# ---------------------------------------------------------------------------
# Chemistry helpers
# ---------------------------------------------------------------------------

def canonicalize_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES string.

    Raises:
        ValueError: If smiles does not parse to a valid molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def validate_smiles(smiles: str) -> bool:
    """Return True if smiles parses to a valid RDKit molecule."""
    return Chem.MolFromSmiles(smiles) is not None


def smiles_to_inchikey(smiles: str) -> str:
    """Convert canonical SMILES to InChIKey.

    Args:
        smiles: Canonical SMILES string.

    Returns:
        InChIKey string (27 characters).

    Raises:
        ValueError: If smiles is invalid or InChI generation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    inchi_str = MolToInchi(mol)
    if inchi_str is None:
        raise ValueError(f"Could not generate InChI for SMILES: {smiles!r}")
    return InchiToInchiKey(inchi_str)


def mol_to_pyg_data(smiles: str) -> "torch_geometric.data.Data":
    """Convert SMILES to PyTorch Geometric Data object.

    Node features (NODE_FEAT_DIM=9): atom type index, degree, formal charge,
    num hydrogens, aromaticity, in-ring, total valence, chirality, no-implicit-H.

    Edge features (EDGE_FEAT_DIM=5): bond order, aromaticity, conjugation,
    in-ring, stereo.

    Args:
        smiles: Canonical SMILES string.

    Returns:
        PyG Data object with x (node feats), edge_index, edge_attr.

    Raises:
        ValueError: If smiles is invalid.
    """
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    atom_features = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=mol.GetNumAtoms())


def _atom_features(atom: Chem.Atom) -> list[float]:
    symbol = atom.GetSymbol()
    atom_type_idx = _ATOM_TYPE_INDEX.get(symbol, _ATOM_TYPE_INDEX["other"])
    return [
        atom_type_idx / len(_ATOM_TYPES),
        atom.GetDegree() / 10.0,
        atom.GetFormalCharge() / 5.0,
        atom.GetTotalNumHs() / 8.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetTotalValence() / 6.0,
        float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
        float(atom.GetNoImplicit()),
    ]


def _bond_features(bond: Chem.Bond) -> list[float]:
    return [
        bond.GetBondTypeAsDouble() / 3.0,
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo() != Chem.BondStereo.STEREONONE),
    ]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamp + module format.

    Args:
        level: Logging level string, e.g. "INFO", "DEBUG", "WARNING".
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    out_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    cfg: dict,
    ema_model: Optional[nn.Module] = None,
) -> Path:
    """Save model, optimizer state, and config to a checkpoint file.

    Args:
        out_dir: Directory to write checkpoint into.
        model: Model whose state_dict to save.
        optimizer: Optimizer whose state_dict to save.
        step: Current training step (used in filename).
        loss: Current validation loss (stored in payload).
        cfg: Config dict to store alongside weights (for config-match validation on load).
        ema_model: Optional EMA model to also checkpoint.

    Returns:
        Path to the saved checkpoint file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"checkpoint_step{step:08d}.pt"
    payload: dict[str, Any] = {
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
    }
    if ema_model is not None:
        payload["ema_model_state"] = ema_model.state_dict()
    torch.save(payload, path)
    logger.info("Saved checkpoint: %s  (step=%d, loss=%.4f)", path, step, loss)
    return path


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[nn.Module] = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load model (and optionally optimizer/EMA) from a checkpoint file.

    Args:
        path: Path to checkpoint file.
        model: Model to load state into.
        optimizer: If provided, optimizer state is also restored.
        ema_model: If provided, EMA model state is also restored.
        device: Map location for torch.load.

    Returns:
        Full checkpoint payload dict (contains 'step', 'loss', 'config').

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if ema_model is not None and "ema_model_state" in payload:
        ema_model.load_state_dict(payload["ema_model_state"])
    logger.info("Loaded checkpoint: %s  (step=%s)", path, payload.get("step", "?"))
    return payload


def latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return path to the latest checkpoint in a directory, or None if empty.

    Args:
        checkpoint_dir: Directory containing checkpoint_step*.pt files.

    Returns:
        Path to the most recent checkpoint, or None if directory is empty.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    return checkpoints[-1] if checkpoints else None


def resume_or_init(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> int:
    """Load the latest checkpoint if one exists, otherwise start from step 0.

    Args:
        checkpoint_dir: Directory to search for checkpoints.
        model: Model to restore.
        optimizer: Optimizer to restore.
        device: Map location for torch.load.

    Returns:
        Starting step number (0 if no checkpoint found).
    """
    ckpt_path = latest_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        logger.info("No checkpoint found in %s — starting from scratch.", checkpoint_dir)
        return 0
    payload = load_checkpoint(ckpt_path, model, optimizer, device=device)
    return int(payload.get("step", 0)) + 1
