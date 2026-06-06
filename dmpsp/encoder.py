"""Molecular encoder: SMILES → dense vector.

Two modes selected via config 'type' key:
  gin      — 3-layer GIN (PyTorch Geometric). Trains from scratch. No external deps.
  chembert — frozen ChemBERTa-77M (HuggingFace). Requires transformers + HF_TOKEN.

Use build_encoder(cfg) to construct the right encoder from a config dict.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from dmpsp.utils import mol_to_pyg_data, NODE_FEAT_DIM

logger = logging.getLogger(__name__)


class MolecularEncoder(ABC, nn.Module):
    """Abstract base for all molecular encoders.

    Subclasses implement encode() which maps a list of SMILES to
    a batch tensor on the given device.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    @abstractmethod
    def encode(self, smiles_list: list[str], device: torch.device) -> torch.Tensor:
        """Encode a batch of SMILES strings into dense vectors.

        Args:
            smiles_list: List of canonical SMILES strings, length B.
            device: Target device for output tensor.

        Returns:
            Tensor of shape (B, hidden_dim).
        """


class GINEncoder(MolecularEncoder):
    """3-layer Graph Isomorphism Network encoder.

    Trained from scratch jointly with ActionProposalDiffusion.
    Frozen for DynamicsDiffusion and ValueFunction (set freeze_encoder=True in config).

    Architecture:
        node_proj → [GINConv → BatchNorm → ReLU → Dropout] × num_layers
        → global mean pool → out_proj
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(hidden_dim)
        try:
            from torch_geometric.data import Batch
            from torch_geometric.nn import GINConv, global_mean_pool
        except Exception as exc:
            raise ImportError(
                f"GINEncoder requires torch_geometric (error: {exc}). "
                "Use encoder type 'morgan' if torch_geometric is unavailable."
            ) from exc
        self._Batch = Batch
        self._global_mean_pool = global_mean_pool

        self.num_layers = num_layers
        self.dropout_p = dropout

        self.node_proj = nn.Linear(NODE_FEAT_DIM, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data: Batch) -> torch.Tensor:
        """Forward pass on a batched PyG graph.

        Args:
            data: Batched PyG Data with attributes: x, edge_index, batch.

        Returns:
            Graph-level embeddings, shape (B, hidden_dim).
        """
        x = self.node_proj(data.x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self._global_mean_pool(x, data.batch)
        return self.out_proj(x)

    def encode(self, smiles_list: list[str], device: torch.device) -> torch.Tensor:
        graphs = [mol_to_pyg_data(s) for s in smiles_list]
        batch = self._Batch.from_data_list(graphs).to(device)
        return self.forward(batch)


class ChemBERTaEncoder(MolecularEncoder):
    """Frozen ChemBERTa-zinc-base-v1 encoder from HuggingFace.

    Always frozen — used as a feature extractor, not fine-tuned.
    Requires the 'transformers' package. Set HF_TOKEN in .env if needed.

    ChemBERTa hidden size (384) is projected to target hidden_dim.
    """

    MODEL_NAME: str = "seyonec/ChemBERTa-zinc-base-v1"

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__(hidden_dim)
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "ChemBERTaEncoder requires the 'transformers' package. "
                "Install with: pip install transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.bert = AutoModel.from_pretrained(self.MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_dim: int = self.bert.config.hidden_size
        self.proj = nn.Linear(bert_dim, hidden_dim)
        logger.info(
            "ChemBERTaEncoder: %s (bert_dim=%d → hidden_dim=%d)",
            self.MODEL_NAME, bert_dim, hidden_dim,
        )

    def encode(self, smiles_list: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = self.bert(**tokens)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.proj(cls_emb)


class MorganFPEncoder(MolecularEncoder):
    """RDKit Morgan fingerprint encoder.

    2048-bit Morgan circular fingerprint (radius=2) → Linear(2048, hidden_dim).
    Fingerprint computation is deterministic (no training needed for the FP itself).
    The linear projection layer is saved with the checkpoint so all models
    trained in the same run share the identical molecular representations.

    No PyG or HuggingFace required — only RDKit (always available).
    """

    FP_DIM: int = 2048

    def __init__(self, hidden_dim: int = 256, radius: int = 2) -> None:
        super().__init__(hidden_dim)
        self.radius = radius
        self.proj = nn.Linear(self.FP_DIM, hidden_dim, bias=True)
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=self.FP_DIM
        )
        self._cache: dict[str, torch.Tensor] = {}  # per-process FP cache

    def _smiles_to_fp(self, smiles: str) -> torch.Tensor:
        if smiles in self._cache:
            return self._cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp_tensor = torch.zeros(self.FP_DIM)
        else:
            bits = self._generator.GetFingerprint(mol)
            fp_tensor = torch.tensor(list(bits), dtype=torch.float)
        self._cache[smiles] = fp_tensor
        return fp_tensor

    def encode(self, smiles_list: list[str], device: torch.device) -> torch.Tensor:
        fps = torch.stack([self._smiles_to_fp(s) for s in smiles_list]).to(device)
        return self.proj(fps.to(next(self.proj.parameters()).device))


_ENCODER_REGISTRY: dict[str, type[MolecularEncoder]] = {
    "gin": GINEncoder,
    "chembert": ChemBERTaEncoder,
    "morgan": MorganFPEncoder,
}


def build_encoder(cfg: dict) -> MolecularEncoder:
    """Construct a MolecularEncoder from a config dict.

    Args:
        cfg: Dict with 'type' key (str) plus encoder-specific kwargs.
             Example: {"type": "gin", "hidden_dim": 256, "num_layers": 3, "dropout": 0.1}

    Returns:
        Constructed MolecularEncoder instance.

    Raises:
        ValueError: If cfg['type'] is not a known encoder type.
    """
    encoder_type = cfg.get("type", "gin")
    if encoder_type not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder type: {encoder_type!r}. "
            f"Choose from: {sorted(_ENCODER_REGISTRY)}"
        )
    encoder_cls = _ENCODER_REGISTRY[encoder_type]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return encoder_cls(**kwargs)
