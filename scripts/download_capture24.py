#!/usr/bin/env python3
"""
CAPTURE-24 Dataset Download Script

This script downloads and prepares the CAPTURE-24 dataset for compression evaluation.
The dataset is available at:
https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001

Note: The dataset is over 6.5GB in size, so download may take some time.
"""

import os
import sys
import requests
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset URL
CAPTURE24_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001"
DATA_DIR = Path("data/capture24")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Size of chunks to download
    """
    logger.info(f"Downloading from {url}")
    logger.info(f"Saving to {output_path}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                bar.update(size)
        
        logger.info(f"Download completed: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        logger.info("\nNote: The CAPTURE-24 dataset may require manual download.")
        logger.info("Please visit: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001")
        logger.info("After downloading, extract the data to: data/capture24/")
        return False


def check_data_availability() -> bool:
    """
    Check if CAPTURE-24 data is already available.
    
    Returns:
        True if data appears to be present
    """
    # Check for common data files
    possible_files = [
        DATA_DIR / "capture24.zip",
        DATA_DIR / "capture24.tar.gz",
        DATA_DIR / "data.csv",
    ]
    
    # Check for participant files
    participant_files = list(DATA_DIR.glob("P*.csv"))
    
    if participant_files:
        logger.info(f"Found {len(participant_files)} participant data files")
        return True
    
    for file_path in possible_files:
        if file_path.exists():
            logger.info(f"Found data archive: {file_path}")
            return True
    
    return False


def main():
    """Main download function."""
    print("CAPTURE-24 Dataset Download")
    print("=" * 60)
    
    # Check if data already exists
    if check_data_availability():
        logger.info("CAPTURE-24 data appears to be already available.")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            logger.info("Skipping download.")
            return
    
    # The actual download URL may require authentication or may be a direct download
    # For now, provide instructions
    logger.info("\nCAPTURE-24 Dataset Download Instructions:")
    logger.info("=" * 60)
    logger.info("1. Visit: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001")
    logger.info("2. Download the dataset (over 6.5GB)")
    logger.info("3. Extract the data to: data/capture24/")
    logger.info("4. Ensure participant data files are in CSV format (P001.csv, P002.csv, etc.)")
    logger.info("\nAlternatively, the repository at https://github.com/OxWearables/capture24")
    logger.info("provides scripts to prepare the data using prepare_data.py")
    logger.info("=" * 60)
    
    # Try to download if URL is direct
    download_path = DATA_DIR / "capture24_data.zip"
    
    try:
        logger.info("\nAttempting direct download...")
        success = download_file(CAPTURE24_URL, download_path)
        
        if success:
            logger.info("\nDownload successful! Please extract the archive to data/capture24/")
            logger.info("You may need to convert the data format to CSV for use with this project.")
        else:
            logger.info("\nDirect download not available. Please follow manual instructions above.")
            
    except Exception as e:
        logger.error(f"Error during download: {e}")
        logger.info("\nPlease download the dataset manually using the instructions above.")


if __name__ == "__main__":
    main()

