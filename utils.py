# Import the math module for mathematical operations (like ceil)
import math
# Import the logging module to record execution events and errors
import logging
# Import hashlib to create unique filenames for cached images
import hashlib
# Import Path to handle file system paths across different operating systems
from pathlib import Path
# Import time and wraps for the retry decorator
import time
from functools import wraps

# Import numpy for efficient array and numerical operations
import numpy as np
# Import requests to handle HTTP GET/POST requests
import requests
# Import pyvo to perform Virtual Observatory (VO) TAP queries
import pyvo

# Import Astropy units to manage physical quantities like degrees or arcminutes
import astropy.units as u
# Import SkyCoord to represent celestial coordinates, and Angle to parse strings
from astropy.coordinates import SkyCoord, Angle
# Import fits module to read and write FITS image files
from astropy.io import fits
# Import Table to manipulate tabular data structures
from astropy.table import Table
# Import Time to calculate current epoch for proper motion adjustments
from astropy.time import Time

# Import MAST Catalogs to query Pan-STARRS and other archives
from astroquery.mast import Catalogs
# Import Gaia module to query the Gaia DR3 database
from astroquery.gaia import Gaia
# Import SkyView to retrieve sky images from NASA's database
from astroquery.skyview import SkyView
# Import Irsa to query the NASA/IPAC Infrared Science Archive (for 2MASS)
from astroquery.irsa import Irsa

# Import Google API modules for Drive upload
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload

# Define a function to initialize and configure a custom logger
def setup_logger(name="myapp", logfile="myapp.log", level=logging.INFO):
    log_path = Path(logfile)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="a")
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Define exponential backoff decorator for network resilience
def retry_with_backoff(retries=5, backoff_in_seconds=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        print(f"❌ Failed after {retries} retries: {e}")
                        raise
                    wait = (backoff_in_seconds * 2 ** x)
                    print(f"⚠️ Query failed ({e}). Retrying in {wait} seconds...")
                    time.sleep(wait)
                    x += 1
        return wrapper
    return decorator

# Define a function to parse string coordinates into float degrees
def parse_coords(ra_str, dec_str):
    ra_str = str(ra_str).strip().lower()
    dec_str = str(dec_str).strip().lower()

    if ':' in ra_str or any(c in ra_str for c in 'hms'):
        ra = Angle(ra_str, unit=u.hourangle)
    else:
        ra = Angle(float(ra_str), unit=u.deg)
    
    if any(c in dec_str for c in 'dms') or ':' in dec_str:
        dec = Angle(dec_str, unit=u.deg)
    else:
        dec = Angle(float(dec_str), unit=u.deg)

    return ra.deg, dec.deg

# ----------------- CACHE SYSTEM -----------------
# Define a robust fetcher that caches FITS files locally to save bandwidth and time
@retry_with_backoff(retries=3)
def fetch_fits_cached(url, cache_dir="./fits_cache"):
    # Ensure cache directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # Generate a unique MD5 hash based on the URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    # Define the local file path for this specific image
    cache_file = Path(cache_dir) / f"{url_hash}.fits"
    
    # Check if the file was already downloaded previously
    if cache_file.exists():
        return fits.open(cache_file)
        
    # If not in cache, execute the HTTP GET request
    response = requests.get(url, timeout=30)
    if response is None or response.status_code != 200:
        return None
        
    # Verify the binary content looks like a FITS file (basic check)
    if not response.content.startswith(b'SIMPLE'):
        return None
        
    # Save the downloaded content to the cache folder
    with open(cache_file, 'wb') as f:
        f.write(response.content)
    
    # Open and return the newly cached FITS file
    return fits.open(cache_file)
# ------------------------------------------------

# Define a function to query reference stars from Gaia DR3
@retry_with_backoff()
def query_stars_gaia(ra, dec, radius=3):
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    # Added pmra and pmdec to the SELECT query
    query = f"""
    SELECT
        source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {(radius*u.arcmin).to(u.deg).value})
    )
    AND ruwe < 1.4
    AND visibility_periods_used > 8
    AND astrometric_excess_noise < 1
    AND phot_bp_rp_excess_factor BETWEEN
        1.0 + 0.015 * POWER(phot_bp_mean_mag - phot_rp_mean_mag, 2)
        AND
        1.3 + 0.06 * POWER(phot_bp_mean_mag - phot_rp_mean_mag, 2)
    ORDER BY phot_g_mean_mag ASC
    """
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    df = results.to_pandas()
    df = df.rename(columns={"phot_g_mean_mag": "mag"})
    
    # Handle missing proper motion values
    df['pmra'] = df['pmra'].fillna(0)
    df['pmdec'] = df['pmdec'].fillna(0)
    
    # Calculate time difference from Gaia DR3 reference epoch (2016.0)
    current_epoch = Time.now().jyear
    dt = current_epoch - 2016.0
    
    # Apply proper motion corrections to coordinates
    # pmra in Gaia is already multiplied by cos(dec), so we divide by it to get the true RA delta
    df['ra'] += (df['pmra'] / np.cos(np.deg2rad(df['dec']))) * dt / 3600000.0
    df['dec'] += df['pmdec'] * dt / 3600000.0
    
    return df

# Define a function to query reference stars from Pan-STARRS 1
@retry_with_backoff()
def query_stars_ps1(ra, dec, radius=3):
    coord = SkyCoord(ra * u.deg, dec * u.deg)
    tbl = Catalogs.query_region(
        coord, radius=radius * u.arcmin, catalog="Panstarrs", table="stack",
        columns=["raMean", "decMean", "gPSFMag", "rPSFMag", "rKronMag", "qualityFlag"]
    )
    tbl = tbl[(tbl["rPSFMag"] < 19) & (tbl["rPSFMag"] > 14) & (abs(tbl["rPSFMag"] - tbl["rKronMag"]) < 0.05) & (tbl["qualityFlag"] < 128)]
    tbl.sort("rPSFMag")
    df = tbl.to_pandas()
    df = df.rename(columns={"raMean": "ra", "decMean": "dec", "gPSFMag": "mag", "rPSFMag": "mag_r"})
    return df

# Define a function to query reference stars from Legacy Survey DR10
@retry_with_backoff()
def query_stars_ls(ra, dec, radius=6):
    tap_url = "https://datalab.noirlab.edu/tap"
    tap_service = pyvo.dal.TAPService(tap_url)
    query = f"""
    SELECT TOP 100 ra, dec, mag_g, mag_r, mag_z FROM ls_dr10.tractor
    WHERE type = 'PSF' AND ra BETWEEN {ra - radius/2/60} AND {ra + radius/2/60}
    AND dec BETWEEN {dec - radius/2/60} AND {dec + radius/2/60} AND mag_r < 18
    """
    job = tap_service.run_async(query, language="ADQL")
    results = job.to_table()
    df = results.to_pandas()
    df = df.rename(columns={"mag_g": "mag"})
    return df

# Define a function to query infrared reference stars from 2MASS
@retry_with_backoff()
def get_stars_2mass(ra, dec, radius=2):
    target = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    tbl = Irsa.query_region(target, radius=radius * u.arcmin, catalog="fp_psc")
    tbl = tbl["ra", "dec", "j_m", "j_cmsig", "ph_qual", "cc_flg"]
    good = np.array([ph_qa[0] in ['A', 'B'] for ph_qa in tbl["ph_qual"]])
    clean = tbl[good]
    df = clean.to_pandas()
    df = df.rename(columns={"j_m": "mag"})
    return df

# Define a helper to populate standard FITS headers
def populate_header(hdu, w_mark, pixscale, imsize, s_name, ra, dec, npixels):
    hdu[0].header['w_mark'] = w_mark
    hdu[0].header['pixscale'] = pixscale
    hdu[0].header['imsize'] = imsize
    hdu[0].header['s_name'] = s_name
    hdu[0].header['ra'] = ra
    hdu[0].header['dec'] = dec
    hdu[0].header['numpix'] = npixels
    return hdu

# Define a function to download an optical FITS image from Pan-STARRS 1
def get_image_ps1(ra, dec, s_name, imsize=6):
    url = f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits?width=500&height=500&fov={imsize/60}&ra={ra}&dec={dec}&hips=CDS/P/PanSTARRS/DR1/r"
    hdu = fetch_fits_cached(url)
    if not hdu or len(hdu) == 0: return ''
    npixels = len(hdu[0].data)
    pixscale = imsize * 60 / npixels
    return populate_header(hdu, 'PS1', pixscale, imsize, s_name, ra, dec, npixels)

# Define a function to download an optical FITS image from Legacy Survey
def get_image_ls(ra, dec, s_name, imsize=6):
    pixscale = 0.26
    numpix = math.ceil(60 * imsize / pixscale)
    url = f"http://legacysurvey.org/viewer/fits-cutout/?ra={ra}&dec={dec}&layer=dr8&pixscale={pixscale}&bands=r&size={numpix}"
    hdu = fetch_fits_cached(url)
    if not hdu or len(hdu) == 0: return ''
    return populate_header(hdu, 'LS', pixscale, imsize, s_name, ra, dec, numpix)

# Define a function to download an optical FITS image from DECaPS
def get_image_decaps(ra, dec, s_name, imsize=6):
    pixscale = 0.26
    numpix = math.ceil(60 * imsize / pixscale)
    url = f"http://legacysurvey.org/viewer/fits-cutout/?layer=decaps2&ra={ra}&dec={dec}&pixscale={pixscale}&bands=r&size={numpix}"
    hdu = fetch_fits_cached(url)
    if not hdu or len(hdu) == 0: return ''
    return populate_header(hdu, 'LS', pixscale, imsize, s_name, ra, dec, numpix)

# Define a function to download an optical FITS image from the Digitized Sky Survey (DSS)
def get_image_dss(ra, dec, s_name, imsize=6):
    url = f"http://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red&r={ra}&dec={dec}&h={imsize}&w={imsize}&e=J2000"
    hdu = fetch_fits_cached(url)
    if not hdu or len(hdu) == 0: return ''
    npixels = len(hdu[0].data)
    pixscale = imsize * 60 / npixels
    return populate_header(hdu, 'DSS', pixscale, imsize, s_name, ra, dec, npixels)

# Define a function to download an infrared FITS image from 2MASS
def get_image_2mass(ra, dec, s_name, imsize=6):
    url = f"https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?Position={ra},{dec}&Survey=2MASS-J&Radius={imsize/60}&Return=FITS"
    hdu = fetch_fits_cached(url)
    
    # Fallback logic if cached/direct download is corrupted
    if not hdu or len(hdu) == 0:
        coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        try:
            images = SkyView.get_images(position=coord, survey=["2MASS-J"], radius=imsize * u.arcmin, pixels=500)
            if not images: return ''
            hdu = images[0]
        except:
            return ''

    npixels = len(hdu[0].data)
    pixscale = imsize * 60 / npixels
    im = hdu[0].data
    
    # Check for empty/corrupt center
    cent, width = int(npixels / 2), int(0.05 * npixels)
    test_slice = slice(cent - width, cent + width)
    if np.isnan(im[test_slice, test_slice].flatten()).all() or (im[test_slice, test_slice].flatten() == 0).all():
        return ''
        
    return populate_header(hdu, '2MASS', pixscale, imsize, s_name, ra, dec, npixels)

# Define a function to sequentially attempt image downloads from optical catalogs
def get_image_fallbacks(ra, dec, s_name, imsize=5):
    
    # Helper function to check if the image has actual data (not just NaNs or zeros)
    def is_valid(hdu):
        if not hdu or len(hdu) == 0: 
            return False
        im = hdu[0].data
        if im is None: 
            return False
        # Check if the entire image is blank (NaNs) or completely black (zeros)
        if np.all(np.isnan(im)) or np.all(im == 0): 
            return False
        # NEW: Check if the image is mostly masked/saturated (e.g., > 90% NaNs)
        if np.isnan(im).sum() / im.size > 0.90:
            print("⚠️ Warning: Image is heavily masked or saturated. Rejecting...")
            return False
        return True

    # Try Legacy Survey (LS)
    hdu = get_image_ls(ra, dec, s_name, imsize=imsize)
    if is_valid(hdu): return hdu
    
    # Try Pan-STARRS 1 (PS1)
    hdu = get_image_ps1(ra, dec, s_name, imsize=imsize)
    if is_valid(hdu): return hdu
        
    # Try DECaPS
    hdu = get_image_decaps(ra, dec, s_name, imsize=imsize)
    if is_valid(hdu): return hdu

    # Try Digitized Sky Survey (DSS) - covers the whole sky
    hdu = get_image_dss(ra, dec, s_name, imsize=imsize)
    if is_valid(hdu): return hdu
    
    # If all fail or return empty images
    raise TypeError("Could not get a valid image with actual data. Tried LS, PS1, DECaPs, and DSS.")
    
# Define a function to automatically upload the generated PDF to Google Drive
def upload_to_drive(file_path, folder_id, credentials_file="drive_credentials.json"):
    """
    Uploads a file to a specific Google Drive folder using a Service Account.
    """
    # Define the required API scopes (permission to read/write files)
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    # Check if the credentials file exists before attempting upload
    if not Path(credentials_file).exists():
        print(f"Warning: Drive credentials '{credentials_file}' not found. Skipping upload.")
        return None

    try:
        # Load the Service Account credentials from the JSON file
        creds = service_account.Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
        # Build the Google Drive API service
        service = build('drive', 'v3', credentials=creds)

        # Extract the file name from the full path
        file_name = Path(file_path).name
        
        # Define the metadata: file name and the target Drive folder ID
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        
        # Specify the file to upload and its MIME type
        media = MediaFileUpload(str(file_path), mimetype='application/pdf', resumable=True)
        
        # Execute the upload request
        print(f"Uploading {file_name} to Google Drive...")
        file = service.files().create(body=file_metadata, media_body=media, fields='id', supportsAllDrives=True).execute()
        
        # Print success message and return the new Google Drive File ID
        print(f"Upload successful! Drive File ID: {file.get('id')}")
        return file.get('id')
        
    except Exception as e:
        # Catch and print any API or network errors
        print(f"Error uploading to Google Drive: {e}")
        return None
