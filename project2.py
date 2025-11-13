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
from astropy.io import fits

# Import project modules
import query
import rgb_plot

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
    output_file : str, optional
        Output filename for RGB plot (default: 'hudf_rgb_mosaic.png')
    
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
        
        # Step 3: Map filters to RGB channels
        logger.info("\n[Step 3] Mapping filters to RGB channels...")
        rgb_files = rgb_plot.map_filters_to_rgb(data_dir=data_dir)
        
        filter_info = {}
        for channel, filepath in rgb_files.items():
            filename = filepath.split('/')[-1]
            # Extract filter name from filename (typically contains filter designation)
            for filt in ['f850lp', 'f814w', 'f775w', 'f606w', 'f555w', 'f435w', 'f475w', 'f450w']:
                if filt in filename.lower():
                    filter_info[channel] = filt.upper()
                    break
        
        logger.info("Filter mapping:")
        logger.info(f"  Red:   {filter_info.get('red', 'Unknown')} - {rgb_files['red']}")
        logger.info(f"  Green: {filter_info.get('green', 'Unknown')} - {rgb_files['green']}")
        logger.info(f"  Blue:  {filter_info.get('blue', 'Unknown')} - {rgb_files['blue']}")
        
        # Step 4: Load FITS images
        logger.info("\n[Step 4] Loading FITS images...")
        red_data, red_wcs, red_header = rgb_plot.load_fits_image(rgb_files['red'])
        green_data, green_wcs, green_header = rgb_plot.load_fits_image(rgb_files['green'])
        blue_data, blue_wcs, blue_header = rgb_plot.load_fits_image(rgb_files['blue'])
        
        # Step 5: Create RGB composite
        logger.info("\n[Step 5] Creating RGB composite image...")
        logger.info("Color scaling: Using percentile method (1-99.5th percentile)")
        logger.info("Stretch function: Asinh stretch for enhanced dynamic range")
        
        # The percentiles are chosen arbitrarily, I think they produced a nice looking image 
        # (Without spending 5 hours testing values)
        rgb_image = rgb_plot.create_rgb_image(
            red_data, green_data, blue_data,
            red_scale=(80, 99.95),
            green_scale=(80, 99.95),
            blue_scale=(80, 99.95),
            stretch='asinh'
        )
        
        # Step 6: Create plot with WCS coordinates
        logger.info("\n[Step 6] Generating plot with equatorial coordinates...")
        fig, ax = rgb_plot.plot_rgb_with_wcs(
            rgb_image, 
            red_wcs,  # Use WCS from red channel
            title='Hubble Ultra Deep Field - ACS/WFC RGB Composite',
            filter_info=filter_info,
            output_file=output_file
        )
        
        logger.info(f"\n[Complete] RGB mosaic saved to: {output_file}")
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        
        # # Display the plot
        # import matplotlib.pyplot as plt
        # plt.show()
        
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
  # Download data and create RGB mosaic
  python project2.py
  
  # Use existing data without downloading
  python project2.py --no-download
  
  # Specify custom output file
  python project2.py --output my_mosaic.png
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