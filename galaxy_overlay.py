"""
Galaxy overlay module for HUDF analysis.

This module provides functionality to:
- Load photometric redshift catalogs
- Overlay galaxy positions on RGB mosaics
- Color-code galaxies by redshift
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from astropy.coordinates import SkyCoord
import astropy.units as u

# Configure logging
logger = logging.getLogger(__name__)


def load_photoz_catalog(catalog_path, n_galaxies=10, random_sample=True, seed=42):
    """
    Load photometric redshift catalog and return galaxy data.
    
    Parameters
    ----------
    catalog_path : str
        Path to the photo-z catalog file
    n_galaxies : int, optional
        Number of galaxies to load (default: 10)
    random_sample : bool, optional
        If True, randomly sample galaxies across catalog (default: True)
    seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    dict
        Dictionary with 'ra', 'dec', 'z_phot' arrays
    
    Raises
    ------
    FileNotFoundError
        If catalog file doesn't exist
    """
    logger.info(f"Loading photo-z catalog from {catalog_path}")
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    
    # Read catalog file - load ALL valid galaxies first
    ra_list = []
    dec_list = []
    z_phot_list = []
    
    try:
        with open(catalog_path, 'r') as f:
            lines = f.readlines()
            logger.info(f"Catalog has {len(lines)} total lines")
            
            data_line_count = 0
            for i, line in enumerate(lines):
                # Skip comment lines (including ## style comments)
                if line.startswith('#') or line.strip() == '':
                    continue
                
                parts = line.split()
                
                # Debug: print first few data lines to understand format
                if data_line_count < 3:
                    logger.info(f"Data line {data_line_count+1}: {len(parts)} columns - {parts}")
                
                if len(parts) >= 4:
                    try:
                        # Format: ID, RA, DEC, zspec, zref
                        # Columns: 0   1    2     3      4
                        ra = float(parts[1])
                        dec = float(parts[2])
                        zspec = float(parts[3])  # Using spectroscopic redshift
                        
                        # Validate RA/Dec ranges
                        if not (0 <= ra <= 360):
                            logger.debug(f"Invalid RA: {ra}")
                            continue
                        if not (-90 <= dec <= 90):
                            logger.debug(f"Invalid Dec: {dec}")
                            continue
                        
                        # Filter out invalid redshifts
                        # Using zspec (spectroscopic redshift) which is more accurate
                        if zspec > 0 and zspec < 15:  # Reasonable z range
                            ra_list.append(ra)
                            dec_list.append(dec)
                            z_phot_list.append(zspec)
                            data_line_count += 1
                            
                            # Don't break early - load ALL galaxies
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing line {i}: {e}")
                        continue
    
    except Exception as e:
        logger.error(f"Error reading catalog: {e}")
        raise
    
    if len(ra_list) == 0:
        logger.error("No valid galaxies found in catalog!")
        logger.error("Please check catalog format. Expected: ID RA Dec z_phot ...")
        raise ValueError("No valid galaxies loaded from catalog")
    
    logger.info(f"Loaded {len(ra_list)} total valid galaxies from catalog")
    
    # Convert to numpy arrays
    ra_array = np.array(ra_list)
    dec_array = np.array(dec_list)
    z_array = np.array(z_phot_list)
    
    # Sample galaxies if we have more than requested
    if random_sample and len(ra_array) > n_galaxies:
        logger.info(f"Randomly sampling {n_galaxies} galaxies from {len(ra_array)} total")
        np.random.seed(seed)
        indices = np.random.choice(len(ra_array), size=n_galaxies, replace=False)
        ra_array = ra_array[indices]
        dec_array = dec_array[indices]
        z_array = z_array[indices]
    else:
        # Just take first n_galaxies
        ra_array = ra_array[:n_galaxies]
        dec_array = dec_array[:n_galaxies]
        z_array = z_array[:n_galaxies]
    
    logger.info(f"Using {len(ra_array)} galaxies for overlay")
    logger.info(f"RA range: {min(ra_array):.4f} to {max(ra_array):.4f}")
    logger.info(f"Dec range: {min(dec_array):.4f} to {max(dec_array):.4f}")
    logger.info(f"Redshift range: {min(z_array):.3f} to {max(z_array):.3f}")
    
    return {
        'ra': ra_array,
        'dec': dec_array,
        'z_phot': z_array
    }


def overlay_galaxies(ax, wcs, galaxy_data, cmap_name='plasma', 
                     marker_size=200, show_labels=True, label_fontsize=8,
                     circle_linewidth=2):
    """
    Overlay galaxy positions on an existing plot with WCS axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object with WCS projection to plot on
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformation
    galaxy_data : dict
        Dictionary with 'ra', 'dec', 'z_phot' arrays
    cmap_name : str, optional
        Name of matplotlib colormap (default: 'plasma')
    marker_size : float, optional
        Size of galaxy markers in pixels (default: 200)
    show_labels : bool, optional
        Whether to show redshift labels (default: True)
    label_fontsize : int, optional
        Font size for labels (default: 8)
    circle_linewidth : float, optional
        Line width for circle markers (default: 2)
    
    Returns
    -------
    matplotlib.colorbar.Colorbar
        Colorbar object for the redshift scale
    
    Raises
    ------
    ValueError
        If galaxy_data is empty or invalid
    """
    if len(galaxy_data['ra']) == 0:
        raise ValueError("No galaxies to overlay")
    
    logger.info(f"Overlaying {len(galaxy_data['ra'])} galaxies")
    
    # Get image bounds
    ny, nx = wcs.array_shape
    logger.info(f"Image shape: {nx} x {ny} pixels")
    
    # Create colormap for redshifts
    norm = Normalize(vmin=galaxy_data['z_phot'].min(), 
                   vmax=galaxy_data['z_phot'].max())
    cmap = cm.get_cmap(cmap_name)
    
    # Track how many galaxies are actually plotted
    plotted_count = 0
    
    # Plot each galaxy
    for i, (ra, dec, z) in enumerate(zip(galaxy_data['ra'], 
                                         galaxy_data['dec'], 
                                         galaxy_data['z_phot'])):
        try:
            # Convert to pixel coordinates
            coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
            x, y = wcs.world_to_pixel(coord)
            
            # Check if galaxy is within image bounds
            if 0 <= x < nx and 0 <= y < ny:
                color = cmap(norm(z))
                
                # Plot HOLLOW circle marker (not filled)
                ax.plot(x, y, 'o', 
                       color='none',  # No fill
                       markeredgecolor=color, 
                       markeredgewidth=circle_linewidth,
                       markersize=marker_size**0.5, 
                       alpha=0.9, 
                       zorder=10)
                
                # Add redshift label if requested
                if show_labels:
                    # Position label outside the circle
                    label_offset = (marker_size**0.5) * 0.7
                    ax.text(x+label_offset, y+label_offset, f'z={z:.2f}', 
                           color='white', 
                           fontsize=label_fontsize,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='black', alpha=0.7),
                           zorder=11)
                
                plotted_count += 1
                logger.debug(f"Galaxy {i+1}: RA={ra:.4f}, Dec={dec:.4f}, "
                           f"z={z:.3f} -> pixel ({x:.1f}, {y:.1f})")
            else:
                logger.debug(f"Galaxy {i+1} outside image bounds: "
                           f"pixel ({x:.1f}, {y:.1f})")
                
        except Exception as e:
            logger.warning(f"Error plotting galaxy {i+1}: {e}")
            continue
    
    logger.info(f"Successfully plotted {plotted_count} out of "
               f"{len(galaxy_data['ra'])} galaxies")
    
    if plotted_count == 0:
        logger.warning("WARNING: No galaxies were within the image boundaries!")
        logger.warning("This might indicate a coordinate mismatch or catalog issue.")
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Photometric Redshift (z)', fontsize=11)
    
    logger.info("Galaxy overlays completed")
    
    return cbar


def create_galaxy_overlay_plot(rgb_image, wcs, catalog_path, n_galaxies=10,
                               title='HUDF with Galaxy Detections',
                               filter_info=None, output_file=None):
    """
    Create a complete plot with RGB image and galaxy overlays.
    
    Parameters
    ----------
    rgb_image : numpy.ndarray
        RGB image array with shape (height, width, 3)
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformation
    catalog_path : str
        Path to photo-z catalog file
    n_galaxies : int, optional
        Number of galaxies to overlay (default: 10)
    title : str, optional
        Plot title (default: 'HUDF with Galaxy Detections')
    filter_info : dict, optional
        Dictionary with filter information for display
    output_file : str, optional
        If provided, save plot to this file
    
    Returns
    -------
    tuple
        (fig, ax, cbar) matplotlib figure, axes, and colorbar objects
    """
    logger.info("Creating galaxy overlay plot")
    
    # Create figure with WCS projection
    fig = plt.figure(figsize=(14, 12))
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
    
    # Load and overlay galaxies
    try:
        galaxy_data = load_photoz_catalog(catalog_path, n_galaxies=n_galaxies)
        cbar = overlay_galaxies(ax, wcs, galaxy_data)
    except Exception as e:
        logger.error(f"Failed to overlay galaxies: {e}")
        raise
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
        logger.info(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved successfully to {output_path}")
    
    return fig, ax, cbar