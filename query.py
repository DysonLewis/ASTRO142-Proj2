"""
Query and download Hubble Ultra Deep Field (HUDF) data from STScI archive.

This module provides functionality to:
- Get HUDF coordinates using SkyCoord
- Download _drz_img.fits files from the HLSP UDF archive
- Download WFC3-IR imaging data
- Download photometric redshift catalog from NED
- Download spectroscopic redshift catalog
"""

import os
import logging
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Get logger (don't configure it - let main script handle that)
logger = logging.getLogger(__name__)


def get_hudf_coordinates():
    """
    Get the coordinates of the Hubble Ultra Deep Field.
    
    Returns
    -------
    astropy.coordinates.SkyCoord
        SkyCoord object with HUDF central coordinates (J2000)
    
    Notes
    -----
    HUDF is centered at RA=03h32m39s, Dec=-27d47m29s (J2000)
    """
    logger.info("Getting HUDF coordinates")
    
    # HUDF central coordinates
    hudf_coord = SkyCoord(ra='03h32m39s', dec='-27d47m29s', frame='icrs')
    
    logger.info(f"HUDF coordinates: RA={hudf_coord.ra.deg:.6f} deg, Dec={hudf_coord.dec.deg:.6f} deg")
    
    return hudf_coord


def download_drz_images(base_url='https://archive.stsci.edu/pub/hlsp/udf/acs-wfc/', 
                        output_dir='./data'):
    """
    Download all _drz_img.fits files from the HUDF ACS-WFC archive.
    
    Parameters
    ----------
    base_url : str, optional
        Base URL of the archive (default: STScI HLSP UDF ACS-WFC)
    output_dir : str, optional
        Directory to save downloaded files (default: './data')
    
    Returns
    -------
    list of str
        List of downloaded file paths
    
    Raises
    ------
    requests.RequestException
        If there's an error accessing the archive
    OSError
        If there's an error creating the output directory
    
    Notes
    -----
    This function scrapes the archive directory listing and downloads
    all files ending with '_drz_img.fits'
    """
    # Input validation
    if not isinstance(base_url, str) or not base_url.startswith('http'):
        raise ValueError(f"Invalid base_url: {base_url}. Must be a valid HTTP(S) URL.")
    
    if not isinstance(output_dir, str):
        raise TypeError(f"output_dir must be a string, got {type(output_dir)}")
    
    logger.info(f"Starting download from {base_url}")
    
    # Create output directory if it doesn't exist
    try:
        # Make path absolute relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory: {e}")
        raise
    
    downloaded_files = []
    
    try:
        # Get the directory listing
        logger.info("Fetching directory listing...")
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
        
        # Parse HTML to find _drz_img.fits files
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')
        
        drz_files = [link.get('href') for link in links 
                     if link.get('href') and link.get('href').endswith('_drz_img.fits')]
        
        if not drz_files:
            logger.warning("No _drz_img.fits files found in the archive")
            return downloaded_files
        
        logger.info(f"Found {len(drz_files)} _drz_img.fits files")
        
        # Download each file
        for filename in drz_files:
            file_url = urljoin(base_url, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(output_path):
                logger.info(f"File already exists, skipping: {filename}")
                downloaded_files.append(output_path)
                continue
            
            logger.info(f"Downloading {filename}...")
            
            try:
                file_response = requests.get(file_url, timeout=60, stream=True)
                file_response.raise_for_status()
                
                # Write file in chunks
                with open(output_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                logger.info(f"Successfully downloaded {filename} ({file_size:.2f} MB)")
                downloaded_files.append(output_path)
                
            except requests.RequestException as e:
                logger.error(f"Failed to download {filename}: {e}")
                continue
        
        logger.info(f"Download complete. {len(downloaded_files)} files downloaded.")
        return downloaded_files
        
    except requests.RequestException as e:
        logger.error(f"Error accessing archive: {e}")
        raise


def download_wfc3ir_images(base_url='https://archive.stsci.edu/pub/hlsp/hudf12/',
                           output_dir='./data'):
    """
    Download WFC3-IR imaging data for HUDF.
    
    Parameters
    ----------
    base_url : str, optional
        Base URL of the WFC3-IR archive
    output_dir : str, optional
        Directory to save downloaded files (default: './data')
    
    Returns
    -------
    list of str
        List of downloaded file paths
    """
    if not isinstance(output_dir, str):
        raise TypeError(f"output_dir must be a string, got {type(output_dir)}")
    
    logger.info(f"Starting WFC3-IR download from {base_url}")
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Files to download
    wfc3_files = [
        'hlsp_hudf12_hst_wfc3ir_udfmain_f105w_v1.0_drz.fits',
        'hlsp_hudf12_hst_wfc3ir_udfmain_f125w_v1.0_drz.fits',
        'hlsp_hudf12_hst_wfc3ir_udfmain_f140w_v1.0_drz.fits',
        'hlsp_hudf12_hst_wfc3ir_udfmain_f160w_v1.0_drz.fits',
    ]
    
    downloaded_files = []
    
    for filename in wfc3_files:
        file_url = urljoin(base_url, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            logger.info(f"File already exists, skipping: {filename}")
            downloaded_files.append(output_path)
            continue
        
        logger.info(f"Downloading {filename}...")
        
        try:
            file_response = requests.get(file_url, timeout=120, stream=True)
            file_response.raise_for_status()
            
            # Write file in chunks
            with open(output_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Successfully downloaded {filename} ({file_size:.2f} MB)")
            downloaded_files.append(output_path)
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {filename}: {e}")
            continue
    
    logger.info(f"WFC3-IR download complete. {len(downloaded_files)} files downloaded.")
    return downloaded_files


def download_photoz_catalog_from_ned(
    ned_refcode_url='https://ned.ipac.caltech.edu/uri/NED::InRefcode/2006AJ....132..926C',
    output_filename='phot_z.csv',
    output_dir='./'):
    """
    Download photometric redshift catalog from NED by reference code.
    
    This function automatically finds the CSV download link on the NED reference page
    and downloads the photometric redshift catalog.
    
    Parameters
    ----------
    ned_refcode_url : str
        URL of the NED reference code page
    output_filename : str, optional
        Name for the output CSV file (default: 'phot_z.csv')
    output_dir : str, optional
        Directory to save catalog (default: './')
    
    Returns
    -------
    str
        Path to downloaded catalog file
    """
    logger.info(f"Accessing NED reference page: {ned_refcode_url}")
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        logger.info(f"Photo-z catalog already exists: {output_filename}")
        return output_path
    
    try:
        # Step 1: Fetch NED reference page
        response = requests.get(ned_refcode_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Step 2: Find the CSV link
        csv_link = None
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text().lower()
            # Look for links with "csv" or "ascii" in the text
            if 'csv' in text or 'ascii' in text or 'download' in text:
                if href.startswith('/cgi-bin/') or href.startswith('https://ned.ipac.caltech.edu/'):
                    csv_link = href
                    break
        
        if csv_link is None:
            raise ValueError("Could not find CSV download link on the NED reference page.")
        
        # Make sure the link is absolute
        if not csv_link.startswith('http'):
            csv_link = f'https://ned.ipac.caltech.edu{csv_link}'
        
        logger.info(f"Found CSV download link: {csv_link}")
        
        # Step 3: Download CSV
        csv_response = requests.get(csv_link, timeout=60)
        csv_response.raise_for_status()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(csv_response.text)
        
        # Verify file
        file_size = os.path.getsize(output_path)
        if file_size < 100:
            logger.warning(f"Downloaded file is very small ({file_size} bytes). Check URL.")
        
        logger.info(f"Successfully downloaded photo-z catalog: {output_filename} ({file_size/1024:.2f} KB)")
        return output_path
    
    except requests.RequestException as e:
        logger.error(f"Failed to download CSV: {e}")
        logger.error("Check your internet connection or the NED page URL.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def download_specz_catalog(catalog_url='https://asd.gsfc.nasa.gov/UVUDF/Rafelski_UDF_speczlist15.txt',
                           output_dir='./'):
    """
    Download spectroscopic redshift catalog for HUDF.
    
    Parameters
    ----------
    catalog_url : str, optional
        URL of the spec-z catalog
        (default: 'https://asd.gsfc.nasa.gov/UVUDF/Rafelski_UDF_speczlist15.txt')
    output_dir : str, optional
        Directory to save catalog (default: './' - working directory)
    
    Returns
    -------
    str
        Path to downloaded catalog file
    """
    logger.info(f"Downloading spec-z catalog from {catalog_url}")
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(catalog_url)
    output_path = os.path.join(output_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        logger.info(f"Spec-z catalog already exists: {filename}")
        return output_path
    
    try:
        response = requests.get(catalog_url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'w') as f:
            f.write(response.text)
        
        logger.info(f"Successfully downloaded spec-z catalog: {filename}")
        return output_path
        
    except requests.RequestException as e:
        logger.error(f"Failed to download spec-z catalog: {e}")
        raise