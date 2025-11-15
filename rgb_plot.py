"""
Create RGB composite images from HUDF ACS-WFC filter data.

This module provides functionality to:
- Load FITS images with WCS information
- Map all 4 filter bandpasses (F850LP, F775W, F606W, F435W)
- Combine F850LP + F775W for red channel
- Apply color scaling using histogram/percentile methods
- Generate RGB composite images with equatorial coordinate axes
"""

import os
import logging
import gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import MinMaxInterval, AsinhStretch, ImageNormalize
from astropy.coordinates import SkyCoord
import astropy.units as u

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
        with fits.open(filepath, memmap=True) as hdul:
            # Use memmap to avoid loading entire file into memory at once
            data = hdul[0].data.copy()  # Make a copy to close the file properly
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
    
    # Handle NaN values - use boolean indexing which is more memory efficient
    finite_mask = np.isfinite(data)
    
    if not np.any(finite_mask):
        logger.warning("No valid data found, returning zeros")
        return np.zeros_like(data, dtype=np.float32)
    
    # Calculate percentile values using only valid data
    # Use float32 to save memory if input is float64
    if data.dtype == np.float64:
        valid_data = data[finite_mask].astype(np.float32)
    else:
        valid_data = data[finite_mask]
    
    vmin = np.percentile(valid_data, lower_percentile)
    vmax = np.percentile(valid_data, upper_percentile)
    
    logger.info(f"Percentile range: {vmin:.2e} to {vmax:.2e}")
    
    # Free memory
    del valid_data
    gc.collect()
    
    # Clip and normalize in-place to save memory
    scaled = np.clip(data, vmin, vmax, dtype=np.float32)
    
    # Avoid division by zero
    if vmax > vmin:
        scaled = (scaled - vmin) / (vmax - vmin)
    else:
        scaled = np.zeros_like(scaled)
    
    # Replace NaNs with 0
    scaled[~finite_mask] = 0.0
    
    return scaled


def create_rgb_image(f850_data, f775_data, f606_data, f435_data,
                     red_scale=(1, 99.5), green_scale=(1, 99.5), blue_scale=(1, 99.5),
                     stretch='asinh'):
    """
    Create an RGB composite image from four filter bands.
    
    This function combines F850LP and F775W for red, F606W for green,
    and F435W for blue to create a full-color composite.
    
    Parameters
    ----------
    f850_data : numpy.ndarray
        Data for F850LP filter (I-band, ~850nm)
    f775_data : numpy.ndarray
        Data for F775W filter (I-band, ~775nm)
    f606_data : numpy.ndarray
        Data for F606W filter (V-band, ~606nm)
    f435_data : numpy.ndarray
        Data for F435W filter (B-band, ~435nm)
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
    if not all(isinstance(d, np.ndarray) for d in [f850_data, f775_data, f606_data, f435_data]):
        raise TypeError("All filter data must be numpy arrays")
    
    if not (f850_data.shape == f775_data.shape == f606_data.shape == f435_data.shape):
        raise ValueError(f"Filter shapes must match: F850={f850_data.shape}, "
                        f"F775={f775_data.shape}, F606={f606_data.shape}, F435={f435_data.shape}")
    
    if stretch not in ['asinh', 'linear', 'sqrt']:
        raise ValueError(f"Invalid stretch: {stretch}. Must be 'asinh', 'linear', or 'sqrt'")
    
    logger.info(f"Creating RGB image from 4 filters with {stretch} stretch")
    
    # Combine F850LP and F775W for red channel (average)
    # Use in-place operations and float32 to save memory
    logger.info("Processing red channel (F850LP + F775W)")
    red_data = (f850_data.astype(np.float32) + f775_data.astype(np.float32)) / 2.0
    red_scaled = scale_image_percentile(red_data, red_scale[0], red_scale[1])
    del red_data  # Free memory immediately
    gc.collect()
    
    # Process green channel
    logger.info("Processing green channel (F606W)")
    green_scaled = scale_image_percentile(f606_data, green_scale[0], green_scale[1])
    
    # Process blue channel
    logger.info("Processing blue channel (F435W)")
    blue_scaled = scale_image_percentile(f435_data, blue_scale[0], blue_scale[1])
    
    # Apply stretch if specified
    if stretch == 'asinh':
        logger.info("Applying asinh stretch")
        # Use in-place operations where possible
        red_scaled = np.arcsinh(red_scaled * 10) / np.arcsinh(10)
        green_scaled = np.arcsinh(green_scaled * 10) / np.arcsinh(10)
        blue_scaled = np.arcsinh(blue_scaled * 10) / np.arcsinh(10)
    elif stretch == 'sqrt':
        logger.info("Applying sqrt stretch")
        red_scaled = np.sqrt(red_scaled)
        green_scaled = np.sqrt(green_scaled)
        blue_scaled = np.sqrt(blue_scaled)
    
    # Stack into RGB array - use float32 to save memory
    logger.info("Stacking channels into RGB array")
    rgb = np.dstack([red_scaled, green_scaled, blue_scaled]).astype(np.float32)
    
    # Clean up individual channels
    del red_scaled, green_scaled, blue_scaled
    gc.collect()
    
    logger.info(f"RGB image created with shape: {rgb.shape}, dtype: {rgb.dtype}")
    
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
        Dictionary with filter information for display
    output_file : str, optional
        If provided, save plot to this file
    
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    
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
    
    # Create figure with WCS projection - use lower DPI for memory efficiency
    fig = plt.figure(figsize=(14, 12), dpi=100)
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
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Plot saved successfully")
    
    # Force garbage collection after plotting
    gc.collect()
    
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
        Dictionary with 'f850lp', 'f775w', 'f606w', 'f435w' keys mapping to file paths
    
    Notes
    -----
    Mapping for HUDF ACS-WFC filters:
    - F850LP (I-band, ~850nm) + F775W (I-band, ~775nm) -> Red channel
    - F606W (V-band, ~606nm) -> Green channel
    - F435W (B-band, ~435nm) -> Blue channel
    
    All four filters are loaded and used to create the RGB composite.
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
    filter_mapping = {
        'f850lp': '_z_',
        'f775w': '_i_', 
        'f606w': '_v_',
        'f435w': '_b_'
    }
    
    filter_files = {}
    
    # Find files for each filter using letter codes
    for filt, letter_code in filter_mapping.items():
        matching = [f for f in fits_files if letter_code in f.lower()]
        if matching:
            filter_files[filt] = os.path.join(data_dir, matching[0])
            logger.info(f"{filt.upper()} ({letter_code}): {matching[0]}")
        else:
            raise ValueError(f"Could not find file for filter {filt.upper()} (looking for '{letter_code}')")
    
    # Check that we have all four filters
    if len(filter_files) != 4:
        missing = set(filter_mapping.keys()) - set(filter_files.keys())
        raise ValueError(f"Could not find files for all filters. Missing: {missing}")
    
    return filter_files