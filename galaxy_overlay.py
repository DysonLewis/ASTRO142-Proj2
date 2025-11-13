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


def load_photoz_catalog(catalog_path, n_galaxies=10):
    """
    Load photometric redshift catalog and return galaxy data.
    
    Parameters
    ----------
    catalog_path : str
        Path to the photo-z catalog file
    n_galaxies : int, optional
        Number of galaxies to load (default: 10)
    
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
    
    # Read catalog file
    ra_list = []
    dec_list = []
    z_phot_list = []
    
    with open(catalog_path, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Typically: ID, RA, Dec, z_phot are first columns
                    ra = float(parts[1])
                    dec = float(parts[2])
                    z_phot = float(parts[3])
                    
                    # Filter out invalid redshifts
                    if z_phot > 0 and z_phot < 15:  # Reasonable z range
                        ra_list.append(ra)
                        dec_list.append(dec)
                        z_phot_list.append(z_phot)
                        
                        if len(ra_list) >= n_galaxies:
                            break
                except (ValueError, IndexError):
                    continue
    
    logger.info(f"Loaded {len(ra_list)} galaxies from catalog")
    
    return {
        'ra': np.array(ra_list),
        'dec': np.array(dec_list),
        'z_phot': np.array(z_phot_list)
    }


def overlay_galaxies(ax, wcs, galaxy_data, cmap_name='plasma', 
                     marker_size=10, show_labels=True, label_fontsize=8):
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
        Size of galaxy markers (default: 10)
    show_labels : bool, optional
        Whether to show redshift labels (default: True)
    label_fontsize : int, optional
        Font size for labels (default: 8)
    
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
    
    # Create colormap for redshifts
    norm = Normalize(vmin=galaxy_data['z_phot'].min(), 
                   vmax=galaxy_data['z_phot'].max())
    cmap = cm.get_cmap(cmap_name)
    
    # Plot each galaxy
    for i, (ra, dec, z) in enumerate(zip(galaxy_data['ra'], 
                                         galaxy_data['dec'], 
                                         galaxy_data['z_phot'])):
        # Convert to pixel coordinates
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        x, y = wcs.world_to_pixel(coord)
        
        color = cmap(norm(z))
        ax.plot(x, y, 'o', color=color, markersize=marker_size, 
               markeredgecolor='white', markeredgewidth=1.5,
               alpha=0.8)
        
        # Add redshift label if requested
        if show_labels:
            ax.text(x+15, y, f'z={z:.2f}', color='white', fontsize=label_fontsize,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Photometric Redshift (z)', fontsize=11)
    
    logger.info("Galaxy overlays added successfully")
    
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
    galaxy_data = load_photoz_catalog(catalog_path, n_galaxies=n_galaxies)
    cbar = overlay_galaxies(ax, wcs, galaxy_data)
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
        logger.info(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved successfully")
    
    return fig, ax, cbar