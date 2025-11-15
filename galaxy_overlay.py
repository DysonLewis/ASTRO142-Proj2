"""
Galaxy overlay module for HUDF analysis.

This module provides functionality to:
- Load photometric redshift catalogs from CSV
- Load spectroscopic redshift catalogs from text files
- Cross-match photometric and spectroscopic samples
- Overlay galaxy positions on RGB mosaics
- Color-code galaxies by redshift and distinguish photo-z vs spec-z
- Create comparison plots of photo-z vs spec-z
"""

import os
import logging
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from astropy.coordinates import SkyCoord
import astropy.units as u

# Configure logging
logger = logging.getLogger(__name__)


def load_photoz_catalog(catalog_path='phot_z.csv', n_galaxies=None):
    """
    Load photometric redshift catalog from CSV file.
    
    Parameters
    ----------
    catalog_path : str, optional
        Path to the photo-z CSV file (default: 'phot_z.csv')
    n_galaxies : int, optional
        Number of galaxies to load (default: None, loads all)
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with 'ra', 'dec', 'z_phot' columns
    
    Raises
    ------
    FileNotFoundError
        If catalog file doesn't exist
    """
    logger.info(f"Loading photometric redshift catalog from {catalog_path}")
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Photometric catalog not found: {catalog_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(catalog_path)
        logger.info(f"Loaded {len(df)} rows from photometric catalog")
        
        # Extract relevant columns
        # Based on the CSV structure: RA is column 4, Dec is column 5, Redshift is column 6
        df_clean = pd.DataFrame({
            'ra': df['RA'],
            'dec': df['Dec'],
            'z_phot': df['Redshift (z)'],
            'object_name': df['Object Name']
        })
        
        # Filter valid redshifts
        df_clean = df_clean[df_clean['z_phot'] > 0].copy()
        df_clean = df_clean[df_clean['z_phot'] < 15].copy()
        
        # Remove NaN values
        df_clean = df_clean.dropna(subset=['ra', 'dec', 'z_phot'])
        
        logger.info(f"After filtering: {len(df_clean)} galaxies with valid photo-z")
        
        if len(df_clean) == 0:
            raise ValueError("No valid galaxies with photometric redshifts found")
        
        if n_galaxies is not None and len(df_clean) > n_galaxies:
            df_clean = df_clean.head(n_galaxies)
            logger.info(f"Limited to first {n_galaxies} galaxies")
        
        logger.info(f"Photo-z range: {df_clean['z_phot'].min():.3f} to {df_clean['z_phot'].max():.3f}")
        logger.info(f"RA range: {df_clean['ra'].min():.4f} to {df_clean['ra'].max():.4f}")
        logger.info(f"Dec range: {df_clean['dec'].min():.4f} to {df_clean['dec'].max():.4f}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error reading photometric catalog: {e}")
        raise


def load_specz_catalog(catalog_path, n_galaxies=None):
    """
    Load spectroscopic redshift catalog from text file.
    
    Parameters
    ----------
    catalog_path : str
        Path to the spec-z catalog file
    n_galaxies : int, optional
        Number of galaxies to load (default: None, loads all)
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with 'ra', 'dec', 'z_spec' columns
    
    Raises
    ------
    FileNotFoundError
        If catalog file doesn't exist
    """
    logger.info(f"Loading spectroscopic redshift catalog from {catalog_path}")
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Spectroscopic catalog not found: {catalog_path}")
    
    # Read catalog file
    ra_list = []
    dec_list = []
    z_spec_list = []
    id_list = []
    
    try:
        with open(catalog_path, 'r') as f:
            lines = f.readlines()
            logger.info(f"Spec-z catalog has {len(lines)} total lines")
            
            for i, line in enumerate(lines):
                # Skip comment lines
                if line.startswith('#') or line.strip() == '':
                    continue
                
                parts = line.split()
                
                if len(parts) >= 4:
                    try:
                        # Format: ID, RA, DEC, zspec, zref
                        obj_id = parts[0]
                        ra = float(parts[1])
                        dec = float(parts[2])
                        zspec = float(parts[3])
                        
                        # Validate RA/Dec ranges
                        if not (0 <= ra <= 360) or not (-90 <= dec <= 90):
                            continue
                        
                        # Filter valid redshifts
                        if zspec > 0 and zspec < 15:
                            id_list.append(obj_id)
                            ra_list.append(ra)
                            dec_list.append(dec)
                            z_spec_list.append(zspec)
                            
                    except (ValueError, IndexError):
                        continue
    
    except Exception as e:
        logger.error(f"Error reading spectroscopic catalog: {e}")
        raise
    
    if len(ra_list) == 0:
        raise ValueError("No valid galaxies loaded from spectroscopic catalog")
    
    logger.info(f"Loaded {len(ra_list)} galaxies from spectroscopic catalog")
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': id_list,
        'ra': ra_list,
        'dec': dec_list,
        'z_spec': z_spec_list
    })
    
    if n_galaxies is not None and len(df) > n_galaxies:
        df = df.head(n_galaxies)
        logger.info(f"Limited to first {n_galaxies} galaxies")
    
    logger.info(f"Spec-z range: {df['z_spec'].min():.3f} to {df['z_spec'].max():.3f}")
    logger.info(f"RA range: {df['ra'].min():.4f} to {df['ra'].max():.4f}")
    logger.info(f"Dec range: {df['dec'].min():.4f} to {df['dec'].max():.4f}")
    
    return df


def cross_match_catalogs(photoz_df, specz_df, match_radius_arcsec=1.0):
    """
    Cross-match photometric and spectroscopic catalogs by position.
    
    Parameters
    ----------
    photoz_df : pandas.DataFrame
        Photometric redshift catalog with 'ra', 'dec', 'z_phot'
    specz_df : pandas.DataFrame
        Spectroscopic redshift catalog with 'ra', 'dec', 'z_spec'
    match_radius_arcsec : float, optional
        Matching radius in arcseconds (default: 1.0)
    
    Returns
    -------
    dict
        Dictionary with:
        - 'matched': DataFrame with both photo-z and spec-z
        - 'photoz_only': DataFrame with only photo-z
        - 'specz_only': DataFrame with only spec-z (no photo-z match)
    """
    logger.info(f"Cross-matching catalogs with {match_radius_arcsec}\" radius")
    
    # Create SkyCoord objects
    photoz_coords = SkyCoord(ra=photoz_df['ra'].values*u.degree, 
                             dec=photoz_df['dec'].values*u.degree)
    specz_coords = SkyCoord(ra=specz_df['ra'].values*u.degree,
                           dec=specz_df['dec'].values*u.degree)
    
    # Find matches: for each specz source, find closest photoz source
    idx, d2d, _ = specz_coords.match_to_catalog_sky(photoz_coords)
    
    # Keep only matches within the radius
    match_mask = d2d < match_radius_arcsec*u.arcsec
    
    n_matches = match_mask.sum()
    logger.info(f"Found {n_matches} matches within {match_radius_arcsec}\"")
    
    if n_matches > 0:
        # Create matched catalog
        matched_specz_indices = np.where(match_mask)[0]
        matched_photoz_indices = idx[match_mask]
        
        matched_data = {
            'ra': photoz_df.iloc[matched_photoz_indices]['ra'].values,
            'dec': photoz_df.iloc[matched_photoz_indices]['dec'].values,
            'z_phot': photoz_df.iloc[matched_photoz_indices]['z_phot'].values,
            'z_spec': specz_df.iloc[matched_specz_indices]['z_spec'].values,
            'separation_arcsec': d2d[match_mask].arcsec
        }
        matched_df = pd.DataFrame(matched_data)
    else:
        matched_df = pd.DataFrame(columns=['ra', 'dec', 'z_phot', 'z_spec', 'separation_arcsec'])
    
    # Identify photo-z only sources (not matched to any spec-z)
    matched_photoz_set = set(idx[match_mask]) if n_matches > 0 else set()
    photoz_only_mask = ~photoz_df.index.isin(matched_photoz_set)
    photoz_only_df = photoz_df[photoz_only_mask].copy()
    
    # Identify spec-z only sources (not matched to photo-z)
    specz_only_mask = ~match_mask
    specz_only_df = specz_df[specz_only_mask].copy()
    
    logger.info(f"Matched sources (have both photo-z and spec-z): {len(matched_df)}")
    logger.info(f"Photo-z only: {len(photoz_only_df)}")
    logger.info(f"Spec-z only (no photo-z match): {len(specz_only_df)}")
    
    return {
        'matched': matched_df,
        'photoz_only': photoz_only_df,
        'specz_only': specz_only_df
    }


def overlay_galaxies_by_type(ax, wcs, matched_df, photoz_only_df, specz_only_df=None,
                             cmap_name='plasma', marker_size=150, show_labels=False):
    """
    Overlay galaxies with different markers for photo-z vs spec-z.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object with WCS projection
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformation
    matched_df : pandas.DataFrame
        Galaxies with both photo-z and spec-z (use spec-z for plotting)
    photoz_only_df : pandas.DataFrame
        Galaxies with only photo-z
    specz_only_df : pandas.DataFrame, optional
        Galaxies with only spec-z (no photo-z counterpart)
    cmap_name : str, optional
        Colormap name (default: 'plasma')
    marker_size : float, optional
        Marker size (default: 150)
    show_labels : bool, optional
        Show redshift labels (default: False)
    
    Returns
    -------
    matplotlib.colorbar.Colorbar
        Colorbar object
    """
    # Get image bounds
    ny, nx = wcs.array_shape
    
    # Combine all redshifts to get global color scale
    all_z = []
    if len(matched_df) > 0:
        all_z.extend(matched_df['z_spec'].values)
    if len(photoz_only_df) > 0:
        all_z.extend(photoz_only_df['z_phot'].values)
    if specz_only_df is not None and len(specz_only_df) > 0:
        all_z.extend(specz_only_df['z_spec'].values)
    
    if len(all_z) == 0:
        logger.warning("No galaxies to plot")
        return None
    
    norm = Normalize(vmin=min(all_z), vmax=max(all_z))
    cmap = cm.get_cmap(cmap_name)
    
    plotted = {'matched': 0, 'photoz_only': 0, 'specz_only': 0}
    
    # Use vectorized operations instead of loops for memory efficiency
    # Plot matched sources (have spec-z) - use CIRCLES
    logger.info(f"Plotting {len(matched_df)} matched sources (circles)")
    if len(matched_df) > 0:
        coords = SkyCoord(ra=matched_df['ra'].values*u.degree, 
                         dec=matched_df['dec'].values*u.degree, frame='icrs')
        x_vals, y_vals = wcs.world_to_pixel(coords)
        
        # Filter to only points within image bounds
        mask = (x_vals >= 0) & (x_vals < nx) & (y_vals >= 0) & (y_vals < ny)
        x_plot = x_vals[mask]
        y_plot = y_vals[mask]
        z_plot = matched_df['z_spec'].values[mask]
        
        # Get colors for all points at once
        colors = cmap(norm(z_plot))
        
        # Single scatter call for all matched galaxies
        ax.scatter(x_plot, y_plot, s=marker_size, facecolors='none', 
                  edgecolors=colors, marker='o', linewidths=2,
                  alpha=0.9, zorder=10)
        
        plotted['matched'] = len(x_plot)
    
    # Plot photo-z only sources - use SQUARES
    logger.info(f"Plotting {len(photoz_only_df)} photo-z only sources (squares)")
    if len(photoz_only_df) > 0:
        coords = SkyCoord(ra=photoz_only_df['ra'].values*u.degree,
                         dec=photoz_only_df['dec'].values*u.degree, frame='icrs')
        x_vals, y_vals = wcs.world_to_pixel(coords)
        
        mask = (x_vals >= 0) & (x_vals < nx) & (y_vals >= 0) & (y_vals < ny)
        x_plot = x_vals[mask]
        y_plot = y_vals[mask]
        z_plot = photoz_only_df['z_phot'].values[mask]
        
        colors = cmap(norm(z_plot))
        
        # Single scatter call for all photo-z only galaxies
        ax.scatter(x_plot, y_plot, s=marker_size, facecolors='none',
                  edgecolors=colors, marker='s', linewidths=2,
                  alpha=0.9, zorder=10)
        
        plotted['photoz_only'] = len(x_plot)
    
    # Plot spec-z only sources (no photo-z) - use TRIANGLES if provided
    if specz_only_df is not None and len(specz_only_df) > 0:
        logger.info(f"Plotting {len(specz_only_df)} spec-z only sources (triangles)")
        coords = SkyCoord(ra=specz_only_df['ra'].values*u.degree,
                         dec=specz_only_df['dec'].values*u.degree, frame='icrs')
        x_vals, y_vals = wcs.world_to_pixel(coords)
        
        mask = (x_vals >= 0) & (x_vals < nx) & (y_vals >= 0) & (y_vals < ny)
        x_plot = x_vals[mask]
        y_plot = y_vals[mask]
        z_plot = specz_only_df['z_spec'].values[mask]
        
        colors = cmap(norm(z_plot))
        
        # Single scatter call for all spec-z only galaxies
        ax.scatter(x_plot, y_plot, s=marker_size, facecolors='none',
                  edgecolors=colors, marker='^', linewidths=2,
                  alpha=0.9, zorder=10)
        
        plotted['specz_only'] = len(x_plot)
    
    logger.info(f"Successfully plotted: {plotted['matched']} matched, "
               f"{plotted['photoz_only']} photo-z only, {plotted['specz_only']} spec-z only")
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Redshift (z)', fontsize=11)
    
    # Force garbage collection to free memory
    gc.collect()
    
    # Add legend with ALL THREE marker types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='gray',
               markeredgewidth=2, markersize=8, 
               label='Matched (have both photo-z & spec-z)'),
        Line2D([0], [0], marker='s', color='w', markeredgecolor='gray',
               markeredgewidth=2, markersize=8, 
               label='Photometric z only'),
        Line2D([0], [0], marker='^', color='w', markeredgecolor='gray',
               markeredgewidth=2, markersize=8, 
               label='Spectroscopic z only')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
             framealpha=0.9)
    
    return cbar


def plot_photoz_vs_specz(matched_df, output_file=None):
    """
    Create scatter plot comparing photometric vs spectroscopic redshifts.
    
    Parameters
    ----------
    matched_df : pandas.DataFrame
        DataFrame with 'z_phot' and 'z_spec' columns
    output_file : str, optional
        Output filename for saving plot
    
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes
    """
    logger.info(f"Creating photo-z vs spec-z comparison plot ({len(matched_df)} sources)")
    
    if len(matched_df) == 0:
        logger.warning("No matched sources for comparison plot")
        return None, None
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    z_phot = matched_df['z_phot'].values
    z_spec = matched_df['z_spec'].values
    
    # Scatter plot
    ax.scatter(z_spec, z_phot, alpha=0.6, s=50, c='blue', edgecolors='black', linewidths=0.5)
    
    # Add 1:1 line
    z_range = [min(z_spec.min(), z_phot.min()), max(z_spec.max(), z_phot.max())]
    ax.plot(z_range, z_range, 'k--', linewidth=2, label='1:1 line')
    
    # Calculate statistics
    residuals = z_phot - z_spec
    bias = np.mean(residuals)
    scatter = np.std(residuals)
    nmad = 1.48 * np.median(np.abs(residuals - np.median(residuals)))
    
    # Add statistics text
    stats_text = f"N = {len(matched_df)}\n"
    stats_text += f"Bias = {bias:.4f}\n"
    stats_text += f"σ = {scatter:.4f}\n"
    stats_text += f"NMAD = {nmad:.4f}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Spectroscopic Redshift (z_spec)', fontsize=14)
    ax.set_ylabel('Photometric Redshift (z_phot)', fontsize=14)
    ax.set_title('Photometric vs Spectroscopic Redshift Comparison', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
        logger.info(f"Saving comparison plot to {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info("Comparison plot saved successfully")
    
    return fig, ax


def create_galaxy_overlay_plot(rgb_image, wcs, photoz_path='phot_z.csv', 
                               specz_path='Rafelski_UDF_speczlist15.txt',
                               n_galaxies=50, title='HUDF with Galaxy Detections',
                               filter_info=None, output_file=None):
    """
    Create a complete plot with RGB image and galaxy overlays.
    Distinguishes between photometric and spectroscopic redshifts.
    
    Parameters
    ----------
    rgb_image : numpy.ndarray
        RGB image array with shape (height, width, 3)
    wcs : astropy.wcs.WCS
        WCS object for coordinate transformation
    photoz_path : str, optional
        Path to photo-z catalog file (default: 'phot_z.csv')
    specz_path : str, optional
        Path to spec-z catalog file (default: 'Rafelski_UDF_speczlist15.txt')
    n_galaxies : int, optional
        Max number of galaxies to overlay from each catalog (default: 50)
    title : str, optional
        Plot title (default: 'HUDF with Galaxy Detections')
    filter_info : dict, optional
        Dictionary with filter information for display
    output_file : str, optional
        If provided, save plot to this file
    
    Returns
    -------
    tuple
        (fig, ax, cbar, cross_match_results) matplotlib figure, axes, 
        colorbar objects, and cross-match dictionary
    """
    logger.info("Creating galaxy overlay plot with photo-z and spec-z distinction")
    
    # Load both catalogs
    try:
        photoz_df = load_photoz_catalog(photoz_path, n_galaxies=n_galaxies)
        specz_df = load_specz_catalog(specz_path, n_galaxies=n_galaxies)
    except Exception as e:
        logger.error(f"Failed to load catalogs: {e}")
        raise
    
    # Cross-match catalogs
    cross_match = cross_match_catalogs(photoz_df, specz_df, match_radius_arcsec=1.0)
    
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
    
    # Overlay galaxies
    cbar = overlay_galaxies_by_type(
        ax, wcs,
        cross_match['matched'],
        cross_match['photoz_only'],
        cross_match['specz_only']
    )
    
    plt.tight_layout()
    # Save if output file specified
    if output_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
        logger.info(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Plot saved successfully to {output_path}")
    
    # Create photo-z vs spec-z comparison plot
    if len(cross_match['matched']) > 0:
        comparison_file = output_file.replace('.png', '_photoz_vs_specz.png') if output_file else None
        plot_photoz_vs_specz(cross_match['matched'], output_file=comparison_file)
    
    return fig, ax, cbar, cross_match