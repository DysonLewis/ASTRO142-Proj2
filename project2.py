"""
Astro 142 Project 2: Hubble Ultra Deep Field Analysis
Main script to orchestrate HUDF data download and RGB image creation.
This script:
1. Downloads HUDF ACS-WFC imaging data from STScI archive
2. Creates an RGB composite mosaic with proper filter mapping
3. Displays the result with WCS equatorial coordinates
Date: 2025-11-12
"""

import logging
import sys
import os
import argparse

# Import project modules
import query

# Get script directory for log file
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, 'project2.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main(download_data=True, data_dir='./data', output_file='hudf_rgb_mosaic.png'):
    """
    Main function to run the HUDF analysis pipeline.
    
    Parameters
    ----------
    download_data : bool, optional
        Whether to download data from archive (default: True)
    data_dir : str, optional
        Directory for data storage (default: './data')
    
    Raises
    ------
    Exception
        If any step of the pipeline fails
    """
    logger.info("="*60)
    logger.info("Starting HUDF Analysis Pipeline")
    logger.info("="*60)

    try:
        # Step 1: Get HUDF coordinates
        logger.info("\n[Step 1] Getting HUDF coordinates...")
        hudf_coords = query.get_hudf_coordinates()
        logger.info(f"HUDF Center: RA={hudf_coords.ra.deg:.6f}°, Dec={hudf_coords.dec.deg:.6f}°")

        # Step 2: Download imaging data (if requested)
        if download_data:
            logger.info("\n[Step 2] Downloading ACS-WFC imaging data...")
            downloaded_files = query.download_drz_images(output_dir=data_dir)
            logger.info(f"Successfully downloaded {len(downloaded_files)} FITS files")
        else:
            logger.info("\n[Step 2] Skipping download (using existing data)")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure data has been downloaded to the correct directory")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HUDF RGB Mosaic Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data
  python project2.py
  
  # Use existing data without downloading
  python project2.py --no-download
        """
    )

    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Skip downloading data (use existing files)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory for data storage (default: ./data)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='hudf_rgb_mosaic.png',
        help='Output filename for RGB mosaic (default: hudf_rgb_mosaic.png)'
    )

    args = parser.parse_args()

    # Run main pipeline
    main(
        download_data=not args.no_download,
        data_dir=args.data_dir,
        output_file=args.output
    )