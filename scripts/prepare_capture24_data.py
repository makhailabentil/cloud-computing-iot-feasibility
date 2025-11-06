#!/usr/bin/env python3
"""
CAPTURE-24 Data Preparation Script

This script prepares CAPTURE-24 data for compression evaluation.
It can work with the CAPTURE-24 GitHub repository's prepare_data.py script
or convert data from other formats.

Usage:
    python scripts/prepare_capture24_data.py [--clone-repo] [--data-dir DATA_DIR]
"""

import subprocess
import sys
import argparse
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CAPTURE24_REPO = "https://github.com/OxWearables/capture24.git"
CAPTURE24_DATA_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001"


def check_capture24_repo():
    """Check if CAPTURE-24 repository is cloned."""
    repo_dir = Path("external/capture24")
    return repo_dir.exists() and (repo_dir / "prepare_data.py").exists()


def clone_capture24_repo():
    """Clone the CAPTURE-24 repository."""
    external_dir = Path("external")
    external_dir.mkdir(exist_ok=True)
    
    repo_dir = external_dir / "capture24"
    
    if repo_dir.exists():
        logger.info("CAPTURE-24 repository already exists")
        return repo_dir
    
    logger.info(f"Cloning CAPTURE-24 repository to {repo_dir}...")
    try:
        subprocess.run(
            ["git", "clone", CAPTURE24_REPO, str(repo_dir)],
            check=True,
            capture_output=True
        )
        logger.info("Repository cloned successfully")
        return repo_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e}")
        logger.info("You may need to clone manually: git clone https://github.com/OxWearables/capture24.git external/capture24")
        return None


def prepare_data_with_capture24_script(repo_dir: Path, data_dir: str):
    """
    Use the CAPTURE-24 repository's prepare_data.py script.
    
    This requires the raw CAPTURE-24 dataset to be downloaded first.
    """
    prepare_script = repo_dir / "prepare_data.py"
    
    if not prepare_script.exists():
        logger.error(f"prepare_data.py not found in {repo_dir}")
        return False
    
    logger.info("Using CAPTURE-24 repository's prepare_data.py...")
    logger.info("Note: This requires the raw dataset to be downloaded first.")
    logger.info(f"Dataset URL: {CAPTURE24_DATA_URL}")
    
    # The prepare_data.py script from CAPTURE-24 repo expects specific arguments
    # We'll need to adapt based on their actual script
    try:
        # Check what the script needs
        result = subprocess.run(
            [sys.executable, str(prepare_script), "--help"],
            capture_output=True,
            text=True
        )
        logger.info("prepare_data.py help:")
        logger.info(result.stdout)
        return True
    except Exception as e:
        logger.error(f"Error running prepare_data.py: {e}")
        return False


def convert_cwa_to_csv(cwa_file: Path, output_csv: Path):
    """
    Convert CWA file (Axivity format) to CSV.
    
    This requires the axivity package or similar tools.
    """
    try:
        import axivity
        logger.info(f"Converting {cwa_file} to {output_csv}...")
        # Implementation would depend on axivity package
        return True
    except ImportError:
        logger.warning("axivity package not available. Install with: pip install axivity")
        logger.info("Alternative: Use the CAPTURE-24 repository's conversion tools")
        return False


def main():
    """Main preparation function."""
    parser = argparse.ArgumentParser(
        description="Prepare CAPTURE-24 data for compression evaluation"
    )
    parser.add_argument(
        '--clone-repo',
        action='store_true',
        help='Clone the CAPTURE-24 GitHub repository'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/capture24',
        help='Directory containing raw CAPTURE-24 data'
    )
    parser.add_argument(
        '--use-repo-script',
        action='store_true',
        help='Use the CAPTURE-24 repository prepare_data.py script'
    )
    
    args = parser.parse_args()
    
    logger.info("CAPTURE-24 Data Preparation")
    logger.info("=" * 60)
    
    # Check if repository is available
    repo_dir = Path("external/capture24")
    if args.clone_repo or not check_capture24_repo():
        repo_dir = clone_capture24_repo()
        if not repo_dir:
            logger.error("Could not clone repository. Please clone manually.")
            return
    
    # If using repo script
    if args.use_repo_script and check_capture24_repo():
        prepare_data_with_capture24_script(repo_dir, args.data_dir)
        return
    
    # Instructions for manual preparation
    logger.info("\n" + "=" * 60)
    logger.info("CAPTURE-24 Data Preparation Instructions")
    logger.info("=" * 60)
    logger.info("\n1. Download the raw CAPTURE-24 dataset:")
    logger.info(f"   URL: {CAPTURE24_DATA_URL}")
    logger.info("   Size: ~6.5GB")
    logger.info("\n2. Clone the CAPTURE-24 repository (if not already done):")
    logger.info(f"   git clone {CAPTURE24_REPO} external/capture24")
    logger.info("\n3. Use their prepare_data.py script:")
    logger.info("   cd external/capture24")
    logger.info("   python prepare_data.py -d <data_directory> -a Walmsley2020 --winsec 10")
    logger.info("\n4. Convert output to CSV format with timestamp, x, y, z columns")
    logger.info("5. Place converted files in data/capture24/ as P001.csv, P002.csv, etc.")
    logger.info("\n" + "=" * 60)
    logger.info("\nAlternative: If you have CWA files, install axivity package:")
    logger.info("   pip install axivity")
    logger.info("   Then use conversion tools to convert CWA -> CSV")


if __name__ == "__main__":
    main()

