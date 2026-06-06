"""Data loaders: raw sources → list of ReactionRecord.

All loaders are lazy (yield records one at a time) and accept explicit paths.
No hardcoded paths, no silent failures.

load_csv()    — user-supplied CSV; column names from data config
load_uspto()  — USPTO-50K / USPTO-FULL pickle/CSV
load_ord()    — Open Reaction Database JSON
load_chembl() — ChEMBL SQLite (for ADMET label augmentation)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Generator, Optional

from dmpsp.state import ReactionRecord
from dmpsp.utils import canonicalize_smiles, validate_smiles

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV loader (user-supplied data — priority 1)
# ---------------------------------------------------------------------------

def load_csv(
    path: Path,
    col_cfg: dict,
    reaction_tokenizer: Optional[object] = None,
) -> Generator[ReactionRecord, None, None]:
    """Load reaction records from a user-supplied CSV file.

    Args:
        path: Path to CSV file.
        col_cfg: Column name mapping dict from configs/data.yaml (csv_columns section).
        reaction_tokenizer: Optional tokenizer to convert reaction_class string → int.
                            If None, csv_columns.reaction_class_id must contain integers.

    Yields:
        ReactionRecord for each valid row. Rows with missing required fields or
        invalid SMILES are logged and skipped, never silently dropped without a log entry.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required column names are missing from the CSV header.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    required_cols = {
        "reactant_smiles": col_cfg["reactant_smiles"],
        "product_smiles": col_cfg["product_smiles"],
        "reaction_class_id": col_cfg["reaction_class_id"],
    }
    optional_cols = {
        k: col_cfg[k] for k in (
            "temperature", "pressure", "time_hours", "solvent", "catalyst",
            "reagent_ratio", "yield_percent", "purity_percent", "cost_usd",
        )
        if k in col_cfg
    }

    total = 0
    skipped = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate that required columns exist in the header
        missing = [v for v in required_cols.values() if v not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Available columns: {reader.fieldnames}. "
                f"Update csv_columns in configs/data.yaml to match your file."
            )

        for row_num, row in enumerate(reader, start=2):  # start=2 (header is row 1)
            total += 1
            record = _parse_csv_row(row, row_num, required_cols, optional_cols)
            if record is None:
                skipped += 1
                continue
            yield record

    logger.info(
        "CSV loader: %d total rows, %d valid, %d skipped (%s)",
        total, total - skipped, skipped, path.name,
    )


def _parse_csv_row(
    row: dict,
    row_num: int,
    required_cols: dict,
    optional_cols: dict,
) -> Optional[ReactionRecord]:
    """Parse a single CSV row into a ReactionRecord. Returns None if invalid."""
    # Required: reactant SMILES
    raw_reactants = row.get(required_cols["reactant_smiles"], "").strip()
    if not raw_reactants:
        logger.debug("Row %d: missing reactant_smiles, skipping.", row_num)
        return None

    # Support comma-separated multi-reactant column
    raw_smiles_list = [s.strip() for s in raw_reactants.split(",") if s.strip()]
    reactant_smiles: list[str] = []
    for smi in raw_smiles_list:
        if not validate_smiles(smi):
            logger.debug("Row %d: invalid reactant SMILES %r, skipping row.", row_num, smi)
            return None
        reactant_smiles.append(canonicalize_smiles(smi))

    # Required: product SMILES
    raw_product = row.get(required_cols["product_smiles"], "").strip()
    if not raw_product or not validate_smiles(raw_product):
        logger.debug("Row %d: missing or invalid product_smiles, skipping.", row_num)
        return None
    product_smiles = canonicalize_smiles(raw_product)

    # Required: reaction class id
    raw_class = row.get(required_cols["reaction_class_id"], "").strip()
    if not raw_class:
        logger.debug("Row %d: missing reaction_class_id, skipping.", row_num)
        return None
    try:
        reaction_class_id = int(raw_class)
    except ValueError:
        logger.debug("Row %d: non-integer reaction_class_id %r, skipping.", row_num, raw_class)
        return None

    # Optional fields
    def _float(key: str) -> Optional[float]:
        col = optional_cols.get(key)
        if col is None:
            return None
        val = row.get(col, "").strip()
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def _str(key: str) -> Optional[str]:
        col = optional_cols.get(key)
        if col is None:
            return None
        val = row.get(col, "").strip()
        return val if val else None

    return ReactionRecord(
        reactant_smiles=reactant_smiles,
        product_smiles=product_smiles,
        reaction_class_id=reaction_class_id,
        source="csv",
        temperature=_float("temperature"),
        pressure=_float("pressure"),
        time_hours=_float("time_hours"),
        solvent=_str("solvent"),
        catalyst=_str("catalyst"),
        reagent_ratio=_float("reagent_ratio"),
        yield_percent=_float("yield_percent"),
        purity_percent=_float("purity_percent"),
        cost_usd=_float("cost_usd"),
    )


# ---------------------------------------------------------------------------
# USPTO loader
# ---------------------------------------------------------------------------

def _strip_atom_maps(smiles: str) -> str:
    """Remove atom map numbers (:N) from a SMILES string via RDKit.

    Falls back to regex stripping if RDKit parse fails.
    """
    try:
        from rdkit.Chem import MolFromSmiles, MolToSmiles
        mol = MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("RDKit parse failed")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return MolToSmiles(mol)
    except Exception:
        import re
        return re.sub(r":\d+", "", smiles)


def load_uspto(path: Path) -> Generator[ReactionRecord, None, None]:
    """Load reaction records from USPTO-50K parquet or CSV.

    Supports two formats (auto-detected from column names):

    HF format (pingzhili/uspto-50k parquet):
        rxn_smiles  — atom-mapped reaction SMILES: 'reactants>>product'
        prod_smiles — clean product SMILES
        class       — integer reaction class

    Legacy CSV format:
        reactants     — dot-separated reactant SMILES
        products      — product SMILES
        reaction_type — integer reaction class

    Args:
        path: Path to USPTO parquet or CSV file.

    Yields:
        ReactionRecord per valid reaction.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"USPTO file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from _load_uspto_parquet(path)
    else:
        yield from _load_uspto_csv(path)


def _load_uspto_parquet(path: Path) -> Generator[ReactionRecord, None, None]:
    """Load USPTO from HF-format parquet (rxn_smiles / prod_smiles / class)."""
    import pandas as pd
    df = pd.read_parquet(path)

    # Filter keep=True if column exists
    if "keep" in df.columns:
        df = df[df["keep"] == True]  # noqa: E712

    total = len(df)
    skipped = 0

    for _, row in df.iterrows():
        rxn_smiles_raw = str(row.get("rxn_smiles", "") or "").strip()
        prod_smiles_raw = str(row.get("prod_smiles", "") or "").strip()
        rxn_class = int(row.get("class", 0) or 0)

        if not rxn_smiles_raw:
            skipped += 1
            continue

        # Parse reactants from atom-mapped reaction SMILES (reactants>>product)
        parts = rxn_smiles_raw.split(">>")
        if len(parts) < 2:
            skipped += 1
            continue

        reactants_part = parts[0]
        reactant_smiles: list[str] = []
        valid = True
        for smi in reactants_part.split("."):
            smi = smi.strip()
            if not smi:
                continue
            clean = _strip_atom_maps(smi)
            if not validate_smiles(clean):
                valid = False
                break
            reactant_smiles.append(canonicalize_smiles(clean))

        if not valid or not reactant_smiles:
            skipped += 1
            continue

        # Product: prefer clean prod_smiles column over parsing rxn_smiles
        if prod_smiles_raw and validate_smiles(prod_smiles_raw):
            product_smiles = canonicalize_smiles(prod_smiles_raw)
        else:
            raw_prod = _strip_atom_maps(parts[-1].strip())
            if not validate_smiles(raw_prod):
                skipped += 1
                continue
            product_smiles = canonicalize_smiles(raw_prod)

        yield ReactionRecord(
            reactant_smiles=reactant_smiles,
            product_smiles=product_smiles,
            reaction_class_id=rxn_class,
            source="uspto",
        )

    logger.info(
        "USPTO parquet loader: %d total, %d valid, %d skipped (%s)",
        total, total - skipped, skipped, path.name,
    )


def _load_uspto_csv(path: Path) -> Generator[ReactionRecord, None, None]:
    """Load USPTO from legacy CSV (reactants / products / reaction_type columns)."""
    total = 0
    skipped = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Auto-detect HF-style CSV (rxn_smiles column) vs legacy
        hf_style = "rxn_smiles" in fieldnames

        for row in reader:
            total += 1

            if hf_style:
                rxn_smiles_raw = row.get("rxn_smiles", "").strip()
                prod_smiles_raw = row.get("prod_smiles", "").strip()
                rxn_class_raw = row.get("class", "0").strip()

                parts = rxn_smiles_raw.split(">>") if rxn_smiles_raw else []
                if len(parts) < 2:
                    skipped += 1
                    continue

                reactants_part = parts[0]
                reactant_smiles: list[str] = []
                valid = True
                for smi in reactants_part.split("."):
                    smi = smi.strip()
                    if not smi:
                        continue
                    clean = _strip_atom_maps(smi)
                    if not validate_smiles(clean):
                        valid = False
                        break
                    reactant_smiles.append(canonicalize_smiles(clean))

                if not valid or not reactant_smiles:
                    skipped += 1
                    continue

                if prod_smiles_raw and validate_smiles(prod_smiles_raw):
                    product_smiles = canonicalize_smiles(prod_smiles_raw)
                else:
                    raw_prod = _strip_atom_maps(parts[-1].strip())
                    if not validate_smiles(raw_prod):
                        skipped += 1
                        continue
                    product_smiles = canonicalize_smiles(raw_prod)

                try:
                    rxn_type = int(rxn_class_raw)
                except ValueError:
                    rxn_type = 0

                yield ReactionRecord(
                    reactant_smiles=reactant_smiles,
                    product_smiles=product_smiles,
                    reaction_class_id=rxn_type,
                    source="uspto",
                )

            else:
                reactants_raw = row.get("reactants", "").strip()
                product_raw = row.get("products", "").strip()
                rxn_type_raw = row.get("reaction_type", "0").strip()

                if not reactants_raw or not product_raw:
                    skipped += 1
                    continue

                reactant_smiles_raw = [s.strip() for s in reactants_raw.split(".") if s.strip()]
                reactant_smiles = []
                valid = True
                for smi in reactant_smiles_raw:
                    if not validate_smiles(smi):
                        valid = False
                        break
                    reactant_smiles.append(canonicalize_smiles(smi))

                if not valid or not validate_smiles(product_raw):
                    skipped += 1
                    continue

                try:
                    rxn_type = int(rxn_type_raw)
                except ValueError:
                    rxn_type = 0

                yield ReactionRecord(
                    reactant_smiles=reactant_smiles,
                    product_smiles=canonicalize_smiles(product_raw),
                    reaction_class_id=rxn_type,
                    source="uspto",
                )

    logger.info(
        "USPTO CSV loader: %d total, %d valid, %d skipped (%s)",
        total, total - skipped, skipped, path.name,
    )


# ---------------------------------------------------------------------------
# ORD loader
# ---------------------------------------------------------------------------

def load_ord(path: Path) -> Generator[ReactionRecord, None, None]:
    """Load reaction records from Open Reaction Database JSON files.

    ORD JSON format: each file is a single Reaction protobuf-exported dict.
    Download from: https://github.com/open-reaction-database/ord-data

    Args:
        path: Path to a directory of ORD .json files, or a single .json file.

    Yields:
        ReactionRecord per valid reaction with conditions.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ORD path not found: {path}")

    json_files = list(path.glob("*.json")) if path.is_dir() else [path]
    if not json_files:
        raise ValueError(f"No .json files found in ORD path: {path}")

    total = 0
    skipped = 0

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse ORD file %s: %s", json_file, exc)
                continue

        reactions = data if isinstance(data, list) else [data]
        for rxn in reactions:
            total += 1
            record = _parse_ord_reaction(rxn)
            if record is None:
                skipped += 1
                continue
            yield record

    logger.info(
        "ORD loader: %d total, %d valid, %d skipped (path=%s)",
        total, total - skipped, skipped, path,
    )


def _parse_ord_reaction(rxn: dict) -> Optional[ReactionRecord]:
    """Parse a single ORD reaction dict. Returns None if insufficient data."""
    # Extract inputs (reactants)
    inputs = rxn.get("inputs", {})
    reactant_smiles: list[str] = []
    for inp_key, inp_val in inputs.items():
        for component in inp_val.get("components", []):
            identifiers = component.get("identifiers", [])
            for ident in identifiers:
                if ident.get("type") == "SMILES":
                    smi = ident.get("value", "").strip()
                    if smi and validate_smiles(smi):
                        reactant_smiles.append(canonicalize_smiles(smi))
                    break

    if not reactant_smiles:
        return None

    # Extract outcome (product)
    outcomes = rxn.get("outcomes", [])
    product_smiles: Optional[str] = None
    yield_percent: Optional[float] = None

    for outcome in outcomes:
        for product in outcome.get("products", []):
            for ident in product.get("identifiers", []):
                if ident.get("type") == "SMILES":
                    smi = ident.get("value", "").strip()
                    if smi and validate_smiles(smi):
                        product_smiles = canonicalize_smiles(smi)
                        break
            measurements = product.get("measurements", [])
            for m in measurements:
                if m.get("type") == "YIELD":
                    pct = m.get("percentage", {}).get("value")
                    if pct is not None:
                        yield_percent = float(pct)
        if product_smiles:
            break

    if product_smiles is None:
        return None

    # Extract conditions
    conditions = rxn.get("conditions", {})
    temp_conditions = conditions.get("temperature", {})
    temp_k: Optional[float] = None
    if "setpoint" in temp_conditions:
        temp_val = temp_conditions["setpoint"].get("value")
        temp_units = temp_conditions["setpoint"].get("units", "CELSIUS")
        if temp_val is not None:
            temp_k = float(temp_val) + 273.15 if temp_units == "CELSIUS" else float(temp_val)

    return ReactionRecord(
        reactant_smiles=reactant_smiles,
        product_smiles=product_smiles,
        reaction_class_id=0,    # ORD doesn't provide USPTO-style reaction classes
        source="ord",
        temperature=temp_k,
        yield_percent=yield_percent,
    )


# ---------------------------------------------------------------------------
# ChEMBL loader (ADMET label augmentation)
# ---------------------------------------------------------------------------

def load_chembl(path: Path) -> Generator[dict, None, None]:
    """Load ADMET data from ChEMBL SQLite database for value function training.

    Downloads: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

    Args:
        path: Path to ChEMBL SQLite .db file.

    Yields:
        Dicts with keys: smiles, assay_type, standard_value, standard_units.

    Raises:
        FileNotFoundError: If path does not exist.
        ImportError: If sqlite3 is not available (stdlib, should always be present).
    """
    import sqlite3

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite file not found: {path}")

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    query = """
        SELECT
            cs.canonical_smiles,
            act.standard_type,
            act.standard_value,
            act.standard_units
        FROM activities act
        JOIN compound_structures cs ON cs.molregno = act.molregno
        WHERE act.standard_value IS NOT NULL
          AND act.standard_relation = '='
          AND cs.canonical_smiles IS NOT NULL
        LIMIT 1000000
    """
    try:
        cursor.execute(query)
        for row in cursor:
            smiles, assay_type, value, units = row
            if not validate_smiles(smiles):
                continue
            yield {
                "smiles": canonicalize_smiles(smiles),
                "assay_type": assay_type,
                "standard_value": float(value),
                "standard_units": units,
            }
    finally:
        conn.close()
