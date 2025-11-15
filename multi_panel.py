"""
Multi-panel visualization module for HUDF analysis.

This module creates a multi-panel figure showing:
- Full HUDF mosaic on the left
- 6 zoomed-in subregions on the right (3 rows x 2 columns)
- Annotations indicating subregion locations on the main panel
- Labels for each panel (a, b, c, d, e, f)

Subregions highlight interesting galaxies with different redshift properties.
"""

import os
import logging
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import project modules
import galaxy_overlay

# Configure logging
logger = logging.getLogger(__name__)


def find_interesting_galaxies(photoz_df, specz_df, cross_match, wcs, n_regions=6):
    """
    Find interesting galaxies to zoom in on.
    
    Strategy:
    - Region 1: ID 20388 (specific galaxy at RA=53.1631779, Dec=-27.8123974)
    - Regions 2-4: galaxies with both photo-z and spec-z (matched) at different redshifts
    - Region 5: highest-z photo-z only galaxy
    - Region 6: highest-z spec-z only galaxy
    
    Parameters
    ----------
    photoz_df : pandas.DataFrame
        Photometric redshift catalog
    specz_df : pandas.DataFrame
        Spectroscopic redshift catalog
    cross_match : dict
        Cross-match results from galaxy_overlay.cross_match_catalogs()
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformation
    n_regions : int, optional
        Number of regions to find (default: 6)
    
    Returns
    -------
    list of dict
        List of region dictionaries with keys:
        - 'ra': Right ascension
        - 'dec': Declination
        - 'z': Redshift value
        - 'type': 'matched', 'photoz_only', or 'specz_only'
        - 'label': Description
        - 'id': Galaxy ID (if available)
    """
    logger.info(f"Finding {n_regions} interesting galaxies for zoom regions")
    
    ny, nx = wcs.array_shape
    regions = []
    
    # Helper function to check if coordinates are within image bounds
    def is_in_bounds(ra, dec, margin=100):
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        x, y = wcs.world_to_pixel(coord)
        return (margin <= x < nx-margin) and (margin <= y < ny-margin)
    
    # Region 1: Search for specific galaxy ID 20388
    logger.info("Searching for galaxy ID 20388...")
    target_ra = 53.1631779
    target_dec = -27.8123974
    
    # Search in all catalogs for this galaxy
    found_target = False
    
    # First check if it's in matched catalog
    matched_df = cross_match['matched']
    if len(matched_df) > 0:
        # Find closest match to target coordinates (within 1 arcsec)
        target_coord = SkyCoord(ra=target_ra*u.degree, dec=target_dec*u.degree, frame='icrs')
        for idx, row in matched_df.iterrows():
            gal_coord = SkyCoord(ra=row['ra']*u.degree, dec=row['dec']*u.degree, frame='icrs')
            sep = target_coord.separation(gal_coord).arcsec
            if sep < 1.0 and is_in_bounds(row['ra'], row['dec']):
                regions.append({
                    'ra': row['ra'],
                    'dec': row['dec'],
                    'z': row['z_spec'],
                    'z_phot': row['z_phot'],
                    'type': 'matched',
                    'label': f"ID 20388: z_spec={row['z_spec']:.3f}, z_phot={row['z_phot']:.3f}",
                    'id': '20388'
                })
                logger.info(f"Region 1: Found ID 20388 in matched catalog at z_spec={row['z_spec']:.3f}")
                found_target = True
                break
    
    # If not in matched, check photo-z only
    if not found_target:
        photoz_only_df = cross_match['photoz_only']
        if len(photoz_only_df) > 0:
            target_coord = SkyCoord(ra=target_ra*u.degree, dec=target_dec*u.degree, frame='icrs')
            for idx, row in photoz_only_df.iterrows():
                gal_coord = SkyCoord(ra=row['ra']*u.degree, dec=row['dec']*u.degree, frame='icrs')
                sep = target_coord.separation(gal_coord).arcsec
                if sep < 1.0 and is_in_bounds(row['ra'], row['dec']):
                    regions.append({
                        'ra': row['ra'],
                        'dec': row['dec'],
                        'z': row['z_phot'],
                        'type': 'photoz_only',
                        'label': f"ID 20388: z_phot={row['z_phot']:.3f}",
                        'id': '20388'
                    })
                    logger.info(f"Region 1: Found ID 20388 in photo-z only at z={row['z_phot']:.3f}")
                    found_target = True
                    break
    
    # If still not found, use the coordinates directly
    if not found_target and is_in_bounds(target_ra, target_dec):
        regions.append({
            'ra': target_ra,
            'dec': target_dec,
            'z': 0.0,  # Unknown redshift
            'type': 'matched',
            'label': f"ID 20388 (RA={target_ra:.4f}, Dec={target_dec:.4f})",
            'id': '20388'
        })
        logger.info(f"Region 1: Using coordinates for ID 20388 (not in catalogs)")
    elif not found_target:
        logger.warning("ID 20388 not found or outside image bounds")
    
    # Regions 2-4: Find 3 matched galaxies with different redshifts
    if len(matched_df) > 0:
        # Filter to galaxies within image bounds (excluding the target if already added)
        in_bounds_matched = matched_df[
            matched_df.apply(lambda row: is_in_bounds(row['ra'], row['dec']), axis=1)
        ].copy()
        
        # Remove target galaxy if it's in the list
        if found_target and len(regions) > 0:
            target_ra_check = regions[0]['ra']
            target_dec_check = regions[0]['dec']
            in_bounds_matched = in_bounds_matched[
                ~((np.abs(in_bounds_matched['ra'] - target_ra_check) < 0.001) & 
                  (np.abs(in_bounds_matched['dec'] - target_dec_check) < 0.001))
            ]
        
        if len(in_bounds_matched) >= 3:
            # Sort by redshift and select galaxies across the redshift range
            in_bounds_matched = in_bounds_matched.sort_values('z_spec')
            
            # Select galaxies at 33%, 50%, 75% percentiles
            indices = [
                int(len(in_bounds_matched) * 0.33),
                int(len(in_bounds_matched) * 0.50),
                int(len(in_bounds_matched) * 0.75)
            ]
            
            for i, idx in enumerate(indices):
                row = in_bounds_matched.iloc[idx]
                regions.append({
                    'ra': row['ra'],
                    'dec': row['dec'],
                    'z': row['z_spec'],
                    'z_phot': row['z_phot'],
                    'type': 'matched',
                    'label': f"Matched: z_spec={row['z_spec']:.3f}, z_phot={row['z_phot']:.3f}"
                })
                logger.info(f"Region {len(regions)}: Matched galaxy at z_spec={row['z_spec']:.3f}")
        else:
            # Add what we can find
            for idx, row in in_bounds_matched.iterrows():
                if len(regions) >= 4:
                    break
                regions.append({
                    'ra': row['ra'],
                    'dec': row['dec'],
                    'z': row['z_spec'],
                    'z_phot': row['z_phot'],
                    'type': 'matched',
                    'label': f"Matched: z_spec={row['z_spec']:.3f}, z_phot={row['z_phot']:.3f}"
                })
    
    # Region 5: Find highest-z photo-z only galaxy
    photoz_only_df = cross_match['photoz_only']
    if len(photoz_only_df) > 0:
        in_bounds_photoz = photoz_only_df[
            photoz_only_df.apply(lambda row: is_in_bounds(row['ra'], row['dec']), axis=1)
        ].copy()
        
        if len(in_bounds_photoz) > 0:
            # Get highest redshift photo-z only galaxy
            high_z_photoz = in_bounds_photoz.nlargest(1, 'z_phot').iloc[0]
            regions.append({
                'ra': high_z_photoz['ra'],
                'dec': high_z_photoz['dec'],
                'z': high_z_photoz['z_phot'],
                'type': 'photoz_only',
                'label': f"Highest z (photo-z only): z={high_z_photoz['z_phot']:.3f}"
            })
            logger.info(f"Region {len(regions)}: Highest z photo-z only at z={high_z_photoz['z_phot']:.3f}")
    
    # Region 6: Find highest-z spec-z only galaxy
    specz_only_df = cross_match['specz_only']
    if len(specz_only_df) > 0:
        in_bounds_specz = specz_only_df[
            specz_only_df.apply(lambda row: is_in_bounds(row['ra'], row['dec']), axis=1)
        ].copy()
        
        if len(in_bounds_specz) > 0:
            # Get highest redshift spec-z only galaxy
            high_z_specz = in_bounds_specz.nlargest(1, 'z_spec').iloc[0]
            regions.append({
                'ra': high_z_specz['ra'],
                'dec': high_z_specz['dec'],
                'z': high_z_specz['z_spec'],
                'type': 'specz_only',
                'label': f"Highest z (spec-z only): z={high_z_specz['z_spec']:.3f}"
            })
            logger.info(f"Region {len(regions)}: Highest z spec-z only at z={high_z_specz['z_spec']:.3f}")
    
    # Fill remaining slots if we don't have 6 yet
    while len(regions) < n_regions:
        if len(matched_df) > 0:
            # Add more matched galaxies
            in_bounds_matched = matched_df[
                matched_df.apply(lambda row: is_in_bounds(row['ra'], row['dec']), axis=1)
            ]
            if len(in_bounds_matched) > len(regions):
                row = in_bounds_matched.iloc[len(regions)]
                regions.append({
                    'ra': row['ra'],
                    'dec': row['dec'],
                    'z': row['z_spec'],
                    'z_phot': row['z_phot'],
                    'type': 'matched',
                    'label': f"Matched: z_spec={row['z_spec']:.3f}"
                })
            else:
                break
        else:
            break
    
    logger.info(f"Found {len(regions)} regions to display")
    return regions


def create_zoom_panel(ax, rgb_image, wcs, region, box_size_arcsec=5.0, 
                     show_crosshair=True):
    """
    Create a zoomed-in view of a specific region.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    rgb_image : numpy.ndarray
        Full RGB image
    wcs : astropy.wcs.WCS
        WCS object
    region : dict
        Region dictionary with 'ra', 'dec', 'z', 'type', 'label'
    box_size_arcsec : float, optional
        Size of zoom box in arcseconds (default: 5.0)
    show_crosshair : bool, optional
        Show crosshair at galaxy center (default: True)
    
    Returns
    -------
    tuple
        (x_min, y_min, width, height) for main panel rectangle
    """
    # Convert RA/Dec to pixel coordinates
    coord = SkyCoord(ra=region['ra']*u.degree, dec=region['dec']*u.degree, frame='icrs')
    x_center, y_center = wcs.world_to_pixel(coord)
    
    # Calculate box size in pixels (use float32 to save memory)
    pixel_scale = np.float32(0.05)  # arcsec/pixel (approximate ACS-WFC)
    half_box_pixels = (box_size_arcsec / 2.0) / pixel_scale
    
    # Define cutout bounds
    ny, nx = rgb_image.shape[:2]
    x_min = max(0, int(x_center - half_box_pixels))
    x_max = min(nx, int(x_center + half_box_pixels))
    y_min = max(0, int(y_center - half_box_pixels))
    y_max = min(ny, int(y_center + half_box_pixels))
    
    # Extract cutout (create a copy to avoid holding reference to full image)
    cutout = rgb_image[y_min:y_max, x_min:x_max].copy()
    
    # Display cutout
    ax.imshow(cutout, origin='lower', interpolation='nearest')
    
    # Add crosshair at center
    if show_crosshair:
        cutout_center_x = x_center - x_min
        cutout_center_y = y_center - y_min
        
        # Determine marker based on type
        if region['type'] == 'matched':
            marker = 'o'
            color = 'cyan'
        elif region['type'] == 'photoz_only':
            marker = 's'
            color = 'yellow'
        else:  # specz_only
            marker = '^'
            color = 'lime'
        
        ax.plot(cutout_center_x, cutout_center_y, marker, 
               markersize=12, markerfacecolor='none',
               markeredgecolor=color, markeredgewidth=2)
    
    # Add label
    label_text = region['label']
    ax.text(0.05, 0.95, label_text, transform=ax.transAxes,
           fontsize=8, verticalalignment='top', color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    # Clean up cutout memory
    del cutout
    
    # Return rectangle parameters for main panel
    return x_min, y_min, x_max - x_min, y_max - y_min


def create_multi_panel_figure(rgb_image, wcs, photoz_path, specz_path,
                              n_galaxies=100, output_file=None):
    """
    Create multi-panel figure with full mosaic and zoom insets.
    
    Parameters
    ----------
    rgb_image : numpy.ndarray
        RGB image array
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformation
    photoz_path : str
        Path to photometric redshift catalog
    specz_path : str
        Path to spectroscopic redshift catalog
    n_galaxies : int, optional
        Number of galaxies to load from each catalog (default: 100)
    output_file : str, optional
        Output filename for saving plot
    
    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes array
    """
    logger.info("Creating multi-panel figure with zoom insets")
    
    # Load catalogs
    photoz_df = galaxy_overlay.load_photoz_catalog(photoz_path, n_galaxies=n_galaxies)
    specz_df = galaxy_overlay.load_specz_catalog(specz_path, n_galaxies=n_galaxies)
    
    # Cross-match catalogs
    cross_match = galaxy_overlay.cross_match_catalogs(photoz_df, specz_df)
    
    # Find interesting galaxies
    regions = find_interesting_galaxies(photoz_df, specz_df, cross_match, wcs, n_regions=6)
    
    # Create figure with custom layout (lower DPI for memory efficiency)
    # Left: full mosaic (wide)
    # Right: 3x2 grid of zoom panels (larger with tighter spacing)
    fig = plt.figure(figsize=(16, 12), dpi=100)
    
    # Create TWO separate gridspecs - one for main panel, one for zoom panels
    # This allows independent control of spacing
    from matplotlib.gridspec import GridSpec
    
    # Overall grid: 2 columns [main | zooms]
    gs_main = GridSpec(1, 2, figure=fig,
                   width_ratios=[2, 1.2],
                   left=0.0002, right=0.9998,
                   wspace=0.02)  # Gap between main and zoom columns
    
    # Subgrid for zoom panels (3 rows x 2 columns)
    gs_zoom = gs_main[0, 1].subgridspec(3, 2, hspace=0.15, wspace=0)
    
    # Main panel (left side)
    ax_main = fig.add_subplot(gs_main[0, 0], projection=wcs)
    
    # Zoom panels (right side, 3 rows x 2 columns)
    zoom_axes = []
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    for i in range(3):
        for j in range(2):
            idx = i * 2 + j
            if idx < len(regions):
                ax = fig.add_subplot(gs_zoom[i, j])
                zoom_axes.append(ax)
    
    # === Main Panel ===
    logger.info("Setting up main panel")
    ax_main.imshow(rgb_image, origin='lower', interpolation='nearest')
    
    # Set up coordinates
    ax_main.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=11)
    ax_main.coords[1].set_axislabel('Declination (J2000)', fontsize=11)
    ax_main.coords[0].set_major_formatter('hh:mm:ss')
    ax_main.coords[1].set_major_formatter('dd:mm:ss')
    ax_main.coords.grid(color='white', alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax_main.set_title('Hubble Ultra Deep Field - Multi-Panel View', 
                     fontsize=14, pad=10)
    
    # Add filter info
    filter_info = {
        'red': 'F850LP + F775W',
        'green': 'F606W',
        'blue': 'F435W'
    }
    info_text = (f"Red: {filter_info['red']}\n"
                f"Green: {filter_info['green']}\n"
                f"Blue: {filter_info['blue']}")
    ax_main.text(0.02, 0.98, info_text, transform=ax_main.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === Zoom Panels ===
    logger.info(f"Creating {len(regions)} zoom panels")
    
    rect_colors = ['cyan', 'yellow', 'lime', 'magenta', 'orange', 'red']
    
    for i, (ax_zoom, region, label) in enumerate(zip(zoom_axes, regions, panel_labels)):
        logger.info(f"Creating panel {label} for {region['type']} galaxy at z={region['z']:.3f}")
        
        # Create zoom panel
        x_min, y_min, width, height = create_zoom_panel(
            ax_zoom, rgb_image, wcs, region,
            box_size_arcsec=20.0, show_crosshair=True
        )
        
        # Add panel label
        ax_zoom.text(0.95, 0.05, label, transform=ax_zoom.transAxes,
                   fontsize=14, fontweight='bold', color='white',
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Draw rectangle on main panel
        color = rect_colors[i % len(rect_colors)]
        
        # Main rectangle
        rect = Rectangle((x_min, y_min), width, height,
                        linewidth=2, edgecolor=color, facecolor='none',
                        linestyle='-', zorder=20)
        ax_main.add_patch(rect)
        
        # Add label on main panel
        ax_main.text(x_min + width/2, y_min + height + 10, label,
                   fontsize=10, fontweight='bold', color=color,
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='black', 
                            alpha=0.7, edgecolor=color, linewidth=1.5))
    
    # Add legend to main panel
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='white',
               markeredgewidth=2, markersize=10, linestyle='None',
               label='Circle: matched (both z types)'),
        Line2D([0], [0], marker='s', color='w', markeredgecolor='white',
               markeredgewidth=2, markersize=10, linestyle='None',
               label='Square: photo-z only'),
        Line2D([0], [0], marker='^', color='w', markeredgecolor='white',
               markeredgewidth=2, markersize=10, linestyle='None',
               label='Triangle: spec-z only')
    ]
    ax_main.legend(handles=legend_elements, loc='lower right', 
                  fontsize=9, framealpha=0.9, title='Galaxy Types')
    
    # Save if output file specified
    if output_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
        logger.info(f"Saving multi-panel figure to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info("Multi-panel figure saved successfully")
    
    # Clean up to free memory
    plt.close(fig)
    gc.collect()
    
    return fig, (ax_main, zoom_axes)