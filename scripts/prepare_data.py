"""Data preparation script: download public datasets and/or preprocess user CSV.

Usage examples:
    # Process user-supplied CSV (highest priority data source)
    python scripts/prepare_data.py --source csv --data_path data/raw/reactions.csv \
        --data_config configs/data.yaml --out_dir data/processed/

    # Download and process USPTO-50K
    python scripts/prepare_data.py --source uspto50k --out_dir data/processed/

    # Download and process ORD
    python scripts/prepare_data.py --source ord --out_dir data/processed/

    # Download ChEMBL SQLite (for ADMET labels)
    python scripts/prepare_data.py --source chembl --out_dir data/processed/
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import yaml

# Ensure project root is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_csv, load_ord, load_uspto
from data.preprocess import (
    build_trajectories,
    save_trajectories,
    split_trajectories,
)
from dmpsp.utils import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataset download URLs
# ---------------------------------------------------------------------------

_USPTO_50K_URL = (
    "https://raw.githubusercontent.com/connorcoley/rexgen_direct/"
    "master/rexgen_direct/data/train.csv"
)
_ORD_RELEASE_URL = (
    "https://github.com/open-reaction-database/ord-data/archive/refs/heads/main.zip"
)
_CHEMBL_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_sqlite.tar.gz"
)


def download_file(url: str, dest: Path) -> Path:
    """Download a file from url to dest. Returns dest path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    logger.info("Download complete: %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess synthesis data for DMPSP training."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["csv", "uspto50k", "ord", "chembl"],
        help="Data source to process.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to input data file (required for --source csv or pre-downloaded files).",
    )
    parser.add_argument(
        "--data_config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data config YAML. Default: configs/data.yaml.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for processed trajectories.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Load data config
    config_path = Path(args.data_config)
    if not config_path.exists():
        parser.error(f"Data config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Source-specific loading
    # ---------------------------------------------------------------------------

    if args.source == "csv":
        if args.data_path is None:
            parser.error("--data_path is required for --source csv")
        logger.info("Loading user CSV: %s", args.data_path)
        records = list(load_csv(args.data_path, data_cfg.get("csv_columns", {})))

    elif args.source == "uspto50k":
        raw_path = args.data_path or out_dir / "raw" / "uspto50k_train.csv"
        if not raw_path.exists():
            logger.info("Downloading USPTO-50K training set...")
            download_file(_USPTO_50K_URL, raw_path)
        logger.info("Loading USPTO-50K: %s", raw_path)
        records = list(load_uspto(raw_path))

    elif args.source == "ord":
        raw_path = args.data_path or out_dir / "raw" / "ord"
        if not raw_path.exists():
            logger.info("ORD path does not exist. Download from: %s", _ORD_RELEASE_URL)
            logger.info("Then point --data_path to the extracted directory.")
            sys.exit(1)
        logger.info("Loading ORD: %s", raw_path)
        records = list(load_ord(raw_path))

    elif args.source == "chembl":
        raw_path = args.data_path or out_dir / "raw" / "chembl.db"
        if not raw_path.exists():
            logger.info("ChEMBL SQLite not found. Download from: %s", _CHEMBL_URL)
            logger.info("Then extract and point --data_path to the .db file.")
            sys.exit(1)
        # ChEMBL is handled separately (yields ADMET dicts, not ReactionRecords)
        logger.info("ChEMBL ADMET augmentation not yet implemented — coming in Phase 3.")
        sys.exit(0)

    else:
        parser.error(f"Unknown source: {args.source}")

    if not records:
        logger.error("No valid records loaded. Check input file and column mappings.")
        sys.exit(1)

    logger.info("Loaded %d valid reaction records.", len(records))

    # ---------------------------------------------------------------------------
    # Build trajectories and split
    # ---------------------------------------------------------------------------

    trajectories = build_trajectories(records, data_cfg, seed=data_cfg.get("split_seed", 42))

    train_frac = data_cfg.get("train_frac", 0.90)
    val_frac = data_cfg.get("val_frac", 0.05)
    test_frac = data_cfg.get("test_frac", 0.05)

    train, val, test = split_trajectories(
        trajectories, train_frac, val_frac, test_frac,
        seed=data_cfg.get("split_seed", 42),
    )

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------

    save_trajectories(train, out_dir / "trajectories_train.pkl")
    save_trajectories(val, out_dir / "trajectories_val.pkl")
    save_trajectories(test, out_dir / "trajectories_test.pkl")

    logger.info(
        "Done. Output: %s  [train=%d, val=%d, test=%d]",
        out_dir, len(train), len(val), len(test),
    )


if __name__ == "__main__":
    main()
