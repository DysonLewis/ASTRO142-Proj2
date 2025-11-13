"""
Astro 142 Project 2: Hubble Ultra Deep Field Analysis

Main script to orchestrate HUDF data download and RGB image creation.

This script:
1. Downloads HUDF ACS-WFC imaging data from STScI archive
2. Downloads WFC3-IR imaging data
3. Downloads photometric redshift catalog
4. Creates an RGB composite mosaic with proper filter mapping
5. Overlays galaxy detections color-coded by redshift
6. Displays the result with WCS equatorial coordinates

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
import galaxy_overlay

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


def main(download_data=True, data_dir='./data', output_file='hudf_rgb_mosaic.png',
         n_galaxies=10):
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
    n_galaxies : int, optional
        Number of galaxies to overlay on plot (default: 10)
    
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
            logger.info("\n[Step 2a] Downloading ACS-WFC imaging data...")
            downloaded_files = query.download_drz_images(output_dir=data_dir)
            logger.info(f"Successfully downloaded {len(downloaded_files)} ACS-WFC FITS files")
            
            logger.info("\n[Step 2b] Downloading WFC3-IR imaging data...")
            wfc3_files = query.download_wfc3ir_images(output_dir=data_dir)
            logger.info(f"Successfully downloaded {len(wfc3_files)} WFC3-IR FITS files")
            
            logger.info("\n[Step 2c] Downloading photo-z catalog...")
            catalog_path = query.download_photoz_catalog(output_dir='./')
            logger.info(f"Successfully downloaded catalog: {catalog_path}")
        else:
            logger.info("\n[Step 2] Skipping download (using existing data)")
            catalog_path = os.path.join(script_dir, 'Rafelski_UDF_speczlist15.txt')
        
        # Step 3: Map filters to RGB channels
        logger.info("\n[Step 3] Mapping all 4 filters (F850LP, F775W, F606W, F435W)...")
        filter_files = rgb_plot.map_filters_to_rgb(data_dir=data_dir)
        
        logger.info("Filter mapping:")
        logger.info(f"  F850LP (I-band): {filter_files['f850lp']}")
        logger.info(f"  F775W (I-band):  {filter_files['f775w']}")
        logger.info(f"  F606W (V-band):  {filter_files['f606w']}")
        logger.info(f"  F435W (B-band):  {filter_files['f435w']}")
        
        # Step 4: Load FITS images for all 4 filters
        logger.info("\n[Step 4] Loading all 4 FITS images...")
        f850_data, f850_wcs, f850_header = rgb_plot.load_fits_image(filter_files['f850lp'])
        f775_data, f775_wcs, f775_header = rgb_plot.load_fits_image(filter_files['f775w'])
        f606_data, f606_wcs, f606_header = rgb_plot.load_fits_image(filter_files['f606w'])
        f435_data, f435_wcs, f435_header = rgb_plot.load_fits_image(filter_files['f435w'])
        
        # Step 5: Create RGB composite from all 4 filters
        logger.info("\n[Step 5] Creating RGB composite image from 4 filters...")
        logger.info("Red channel: F850LP + F775W combined")
        logger.info("Green channel: F606W")
        logger.info("Blue channel: F435W")
        logger.info("Color scaling: Using percentile method (80-99.95th percentile)")
        logger.info("Stretch function: Asinh stretch for enhanced dynamic range")
        
        rgb_image = rgb_plot.create_rgb_image(
            f850_data, f775_data, f606_data, f435_data,
            red_scale=(80, 99.95),
            green_scale=(80, 99.95),
            blue_scale=(80, 99.95),
            stretch='asinh'
        )
        
        # Step 6: Create base plot without galaxy overlays
        logger.info("\n[Step 6] Generating base RGB plot with equatorial coordinates...")
        
        filter_info = {
            'red': 'F850LP + F775W',
            'green': 'F606W',
            'blue': 'F435W'
        }
        
        fig, ax = rgb_plot.plot_rgb_with_wcs(
            rgb_image, 
            f850_wcs,  # Use WCS from one of the images
            title='Hubble Ultra Deep Field - ACS/WFC RGB Composite',
            filter_info=filter_info,
            output_file=output_file
        )
        
        # Step 7: Create version with galaxy overlays
        logger.info("\n[Step 7] Creating version with galaxy overlays...")
        logger.info(f"Overlaying {n_galaxies} galaxies from photo-z catalog")
        
        overlay_output = output_file.replace('.png', '_with_galaxies.png')
        
        fig_gal, ax_gal, cbar = galaxy_overlay.create_galaxy_overlay_plot(
            rgb_image,
            f850_wcs,
            catalog_path=catalog_path,
            n_galaxies=n_galaxies,
            title='Hubble Ultra Deep Field - Galaxy Detections by Redshift',
            filter_info=filter_info,
            output_file=overlay_output
        )
        
        logger.info(f"\n[Complete] RGB mosaic saved to: {output_file}")
        logger.info(f"[Complete] Galaxy overlay version saved to: {overlay_output}")
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        
        # # Uncomment to display the plot
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
        description='HUDF RGB Mosaic Generator with Galaxy Overlays',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data and create RGB mosaic with 10 galaxies
  python project2.py
  
  # Use existing data without downloading
  python project2.py --no-download
  
  # Overlay 25 galaxies instead of default 10
  python project2.py --n-galaxies 25
  
  # Specify custom output file
  python project2.py --output my_mosaic.png
  
Output:
  Creates two files:
  1. Base RGB mosaic (specified by --output)
  2. Version with galaxy overlays (*_with_galaxies.png)
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
    
    parser.add_argument(
        '--n-galaxies',
        type=int,
        default=10,
        help='Number of galaxies to overlay on plot (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Run main pipeline
    main(
        download_data=not args.no_download,
        data_dir=args.data_dir,
        output_file=args.output,
        n_galaxies=args.n_galaxies
    )