#!/usr/bin/env python3
"""
Script to download and prepare real CAPTURE-24 data for compression evaluation.

This script:
1. Uses the CAPTURE-24 repository's prepare_data.py to download and extract data
2. Converts the CSV.gz files to plain CSV format for our loader
3. Places them in data/capture24/ with proper naming
"""

import sys
import subprocess
import gzip
import shutil
from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CAPTURE24_REPO_DIR = Path("external/capture24")
CAPTURE24_DATA_DIR = Path("external/capture24/data")
OUR_DATA_DIR = Path("data/capture24")


def download_capture24_data():
    """Download CAPTURE-24 data using their prepare_data.py script."""
    prepare_script = CAPTURE24_REPO_DIR / "prepare_data.py"
    
    if not prepare_script.exists():
        logger.error(f"prepare_data.py not found at {prepare_script}")
        logger.info("Make sure you've cloned the repository: git clone https://github.com/OxWearables/capture24.git external/capture24")
        return False
    
    logger.info("Downloading CAPTURE-24 data (this may take a while, ~6.5GB)...")
    logger.info("This will use the CAPTURE-24 repository's prepare_data.py script")
    
    # Change to the repository directory to run their script
    import os
    original_dir = os.getcwd()
    
    try:
        os.chdir(CAPTURE24_REPO_DIR)
        
        # Run their download function (we'll call it directly)
        # Actually, we can just import and call it
        logger.info("Note: The download will happen automatically when we access the data files")
        logger.info("The prepare_data.py script downloads to data/capture24/ in the repo directory")
        
    finally:
        os.chdir(original_dir)
    
    return True


def convert_gz_to_csv():
    """Convert CSV.gz files from CAPTURE-24 to plain CSV format."""
    # CAPTURE-24 stores files as P001.csv.gz, P002.csv.gz, etc.
    source_dir = CAPTURE24_DATA_DIR / "capture24"
    
    if not source_dir.exists():
        logger.warning(f"Source directory {source_dir} does not exist")
        logger.info("Attempting to download data first...")
        # Try to trigger download by importing their prepare_data
        try:
            sys.path.insert(0, str(CAPTURE24_REPO_DIR))
            from prepare_data import download_capture24
            download_capture24(str(CAPTURE24_DATA_DIR), overwrite=False)
            source_dir = CAPTURE24_DATA_DIR / "capture24"
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            logger.info("\nManual download instructions:")
            logger.info("1. Visit: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001")
            logger.info("2. Download capture24.zip (~6.5GB)")
            logger.info(f"3. Extract to: {CAPTURE24_DATA_DIR}")
            return False
    
    # Ensure output directory exists
    OUR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV.gz files
    gz_files = list(source_dir.glob("P[0-9][0-9][0-9].csv.gz"))
    
    if not gz_files:
        logger.error(f"No CSV.gz files found in {source_dir}")
        logger.info("The dataset may not be downloaded yet")
        return False
    
    logger.info(f"Found {len(gz_files)} participant files to convert")
    
    # Convert each file
    converted_count = 0
    for gz_file in gz_files:
        participant_id = gz_file.stem.replace('.csv', '')  # P001.csv.gz -> P001
        output_file = OUR_DATA_DIR / f"{participant_id}.csv"
        
        # Skip if already exists and is recent
        if output_file.exists():
            logger.info(f"Skipping {participant_id} (already converted)")
            continue
        
        logger.info(f"Converting {participant_id}...")
        
        try:
            # Read the gzipped CSV
            with gzip.open(gz_file, 'rt') as f_in:
                df = pd.read_csv(f_in, index_col='time', parse_dates=['time'])
            
            # Convert time index to timestamp column
            df = df.reset_index()
            df['timestamp'] = df['time'].astype(str)
            
            # Keep only x, y, z columns (and timestamp)
            df = df[['timestamp', 'x', 'y', 'z']]
            
            # Save as plain CSV
            df.to_csv(output_file, index=False)
            logger.info(f"  Saved {len(df)} rows to {output_file}")
            converted_count += 1
            
        except Exception as e:
            logger.error(f"Error converting {participant_id}: {e}")
            continue
    
    logger.info(f"\nConverted {converted_count} participant files")
    logger.info(f"Files are now in: {OUR_DATA_DIR}")
    return converted_count > 0


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("CAPTURE-24 Real Data Preparation")
    logger.info("=" * 60)
    
    # Step 1: Ensure data is downloaded
    logger.info("\nStep 1: Checking for CAPTURE-24 data...")
    if not convert_gz_to_csv():
        logger.error("\nFailed to prepare data. Please ensure:")
        logger.error("1. The CAPTURE-24 repository is cloned in external/capture24/")
        logger.error("2. The dataset is downloaded (run prepare_data.py or download manually)")
        logger.error("3. The data is extracted to external/capture24/data/capture24/")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("Data preparation complete!")
    logger.info(f"Real CAPTURE-24 data is now available in: {OUR_DATA_DIR}")
    logger.info("\nYou can now run:")
    logger.info("  python scripts/evaluate_capture24.py --participants P001,P002 --max-segments 10")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

