"""
Create RGB composite images from HUDF ACS-WFC filter data.

This module provides functionality to:
- Load FITS images with WCS information
- Map filter bandpasses to RGB channels
- Apply color scaling using histogram/percentile methods
- Generate RGB composite images with equatorial coordinate axes
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import MinMaxInterval, AsinhStretch, ImageNormalize

# Configure logging
logger = logging.getLogger(__name__)


def load_fits_image(filepath):
    """
    Load a FITS image file and extract data and WCS.
    
    Parameters
    ----------
    filepath : str
        Path to the FITS file
    
    Returns
    -------
    tuple
        (data, wcs, header) where data is numpy array, wcs is WCS object,
        and header is the FITS header
    
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If the file cannot be read as FITS
    """
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, got {type(filepath)}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading FITS file: {filepath}")
    
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            wcs = WCS(header)
            
            if data is None:
                raise ValueError(f"No data found in {filepath}")
            
            logger.info(f"Loaded image shape: {data.shape}")
            return data, wcs, header
            
    except Exception as e:
        logger.error(f"Error loading FITS file: {e}")
        raise ValueError(f"Failed to read FITS file {filepath}: {e}")


def scale_image_percentile(data, lower_percentile=1, upper_percentile=99.5):
    """
    Scale image data using percentile clipping.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input image data
    lower_percentile : float, optional
        Lower percentile for clipping (default: 1)
    upper_percentile : float, optional
        Upper percentile for clipping (default: 99.5)
    
    Returns
    -------
    numpy.ndarray
        Scaled image data normalized to [0, 1]
    
    Raises
    ------
    TypeError
        If data is not a numpy array
    ValueError
        If percentiles are invalid
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a numpy array, got {type(data)}")
    
    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError(f"Invalid percentiles: lower={lower_percentile}, upper={upper_percentile}")
    
    logger.info(f"Scaling image with percentiles: {lower_percentile}-{upper_percentile}")
    
    # Handle NaN values
    valid_data = data[np.isfinite(data)]
    
    if len(valid_data) == 0:
        logger.warning("No valid data found, returning zeros")
        return np.zeros_like(data)
    
    # Calculate percentile values
    vmin = np.percentile(valid_data, lower_percentile)
    vmax = np.percentile(valid_data, upper_percentile)
    
    logger.info(f"Percentile range: {vmin:.2e} to {vmax:.2e}")
    
    # Clip and normalize
    scaled = np.clip(data, vmin, vmax)
    scaled = (scaled - vmin) / (vmax - vmin)
    
    # Replace NaNs with 0
    scaled = np.nan_to_num(scaled, nan=0.0)
    
    return scaled


def create_rgb_image(red_data, green_data, blue_data, 
                     red_scale=(1, 99.5), green_scale=(1, 99.5), blue_scale=(1, 99.5),
                     stretch='asinh'):
    """
    Create an RGB composite image from three filter bands.
    
    Parameters
    ----------
    red_data : numpy.ndarray
        Data for red channel
    green_data : numpy.ndarray
        Data for green channel
    blue_data : numpy.ndarray
        Data for blue channel
    red_scale : tuple, optional
        (lower, upper) percentiles for red channel (default: (1, 99.5))
    green_scale : tuple, optional
        (lower, upper) percentiles for green channel (default: (1, 99.5))
    blue_scale : tuple, optional
        (lower, upper) percentiles for blue channel (default: (1, 99.5))
    stretch : str, optional
        Stretch function: 'asinh', 'linear', or 'sqrt' (default: 'asinh')
    
    Returns
    -------
    numpy.ndarray
        RGB image with shape (height, width, 3) and values in [0, 1]
    
    Raises
    ------
    ValueError
        If input arrays have different shapes or invalid parameters
    """
    # Input validation
    if not all(isinstance(d, np.ndarray) for d in [red_data, green_data, blue_data]):
        raise TypeError("All channel data must be numpy arrays")
    
    if not (red_data.shape == green_data.shape == blue_data.shape):
        raise ValueError(f"Channel shapes must match: R={red_data.shape}, "
                        f"G={green_data.shape}, B={blue_data.shape}")
    
    if stretch not in ['asinh', 'linear', 'sqrt']:
        raise ValueError(f"Invalid stretch: {stretch}. Must be 'asinh', 'linear', or 'sqrt'")
    
    logger.info(f"Creating RGB image with {stretch} stretch")
    
    # Scale each channel
    red_scaled = scale_image_percentile(red_data, red_scale[0], red_scale[1])
    green_scaled = scale_image_percentile(green_data, green_scale[0], green_scale[1])
    blue_scaled = scale_image_percentile(blue_data, blue_scale[0], blue_scale[1])
    
    # Apply stretch if specified
    if stretch == 'asinh':
        red_scaled = np.arcsinh(red_scaled * 10) / np.arcsinh(10)
        green_scaled = np.arcsinh(green_scaled * 10) / np.arcsinh(10)
        blue_scaled = np.arcsinh(blue_scaled * 10) / np.arcsinh(10)
    elif stretch == 'sqrt':
        red_scaled = np.sqrt(red_scaled)
        green_scaled = np.sqrt(green_scaled)
        blue_scaled = np.sqrt(blue_scaled)
    
    # Stack into RGB array
    rgb = np.dstack([red_scaled, green_scaled, blue_scaled])
    
    logger.info(f"RGB image created with shape: {rgb.shape}")
    
    return rgb


def plot_rgb_with_wcs(rgb_image, wcs, title='HUDF RGB Composite', 
                      filter_info=None, output_file=None):
    """
    Plot RGB image with WCS coordinate axes.
    
    Parameters
    ----------
    rgb_image : numpy.ndarray
        RGB image array with shape (height, width, 3)
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformation
    title : str, optional
        Plot title (default: 'HUDF RGB Composite')
    filter_info : dict, optional
        Dictionary with 'red', 'green', 'blue' keys containing filter names
    output_file : str, optional
        If provided, save plot to this file
    
    Raises
    ------
    ValueError
        If RGB image has wrong shape or WCS is invalid
    """
    if not isinstance(rgb_image, np.ndarray) or rgb_image.ndim != 3:
        raise ValueError(f"rgb_image must be 3D array, got shape {rgb_image.shape}")
    
    if rgb_image.shape[2] != 3:
        raise ValueError(f"rgb_image must have 3 channels, got {rgb_image.shape[2]}")
    
    if not isinstance(wcs, WCS):
        raise TypeError(f"wcs must be WCS object, got {type(wcs)}")
    
    logger.info("Plotting RGB image with WCS coordinates")
    
    # Create figure with WCS projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=wcs)
    
    # Display RGB image
    ax.imshow(rgb_image, origin='lower', interpolation='nearest')
    
    # Set up coordinate labels
    ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=12)
    ax.coords[1].set_axislabel('Declination (J2000)', fontsize=12)
    ax.coords[0].set_major_formatter('hh:mm:ss')
    ax.coords[1].set_major_formatter('dd:mm:ss')
    
    # Add grid
    ax.coords.grid(color='white', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Title
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add filter mapping information
    if filter_info:
        info_text = (f"Red: {filter_info.get('red', 'N/A')}\n"
                    f"Green: {filter_info.get('green', 'N/A')}\n"
                    f"Blue: {filter_info.get('blue', 'N/A')}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        # Make sure output file is saved in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
        logger.info(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved successfully")
    
    return fig, ax


def map_filters_to_rgb(data_dir='./data'):
    """
    Map ACS-WFC filter files to RGB channels.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing FITS files (default: './data')
    
    Returns
    -------
    dict
        Dictionary with 'red', 'green', 'blue' keys mapping to file paths
    
    Notes
    -----
    Standard mapping for HUDF ACS-WFC filters:
    - Red:   F850LP (I-band, ~850nm) 
    - Green: F606W (V-band, ~606nm)
    - Blue:  F435W (B-band, ~435nm)
    
    Files are identified by band designation in filename:
    '_b_' for blue, '_v_' for green/visual, '_i_' for red/infrared or F475W (g-band, ~475nm)
    """
    if not isinstance(data_dir, str):
        raise TypeError(f"data_dir must be a string, got {type(data_dir)}")
    
    # Make path absolute relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, data_dir)
    
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    logger.info(f"Mapping filters from directory: {data_dir}")
    
    # Get all FITS files
    fits_files = [f for f in os.listdir(data_dir) if f.endswith('_drz_img.fits')]
    
    if not fits_files:
        raise ValueError(f"No _drz_img.fits files found in {data_dir}")
    
    logger.info(f"Found {len(fits_files)} FITS files")
    
    # Define filter mapping
    # HUDF ACS-WFC filters:
    # Red: F850LP (I-band, ~850nm)
    # Green: F606W (V-band, ~606nm)
    # Blue: F435W (B-band, ~435nm)
    filter_map = {
        'red': ['f850lp', '_i_'],
        'green': ['f606w', '_v_'],
        'blue': ['f435w', '_b_']
    }
    
    rgb_files = {}
    
    # Find files for each channel
    for channel, filters in filter_map.items():
        for filt in filters:
            matching = [f for f in fits_files if filt in f.lower()]
            if matching:
                rgb_files[channel] = os.path.join(data_dir, matching[0])
                logger.info(f"{channel.upper()} channel: {matching[0]} (filter: {filt.upper()})")
                break
    
    # Check that we have all three channels
    if len(rgb_files) != 3:
        missing = set(['red', 'green', 'blue']) - set(rgb_files.keys())
        raise ValueError(f"Could not find files for all RGB channels. Missing: {missing}")
    
    return rgb_files