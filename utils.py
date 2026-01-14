# ------------------
# Standard library
# ------------------
import io
import math
import re
from io import BytesIO

# ------------------
# Third-party
# ------------------
import numpy as np
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------
# Astropy
# ------------------
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, ZScaleInterval

# ------------------
# Astroquery
# ------------------
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia

# ------------------
# Reproject
# ------------------
from reproject import reproject_interp

# ------------------
# Matplotlib defaults
# ------------------
mpl.rcParams["font.size"] = 15  # default = 10.0



def parse_coords(ra_str, dec_str):
    ra_str = ra_str.strip().lower()
    dec_str = dec_str.strip().lower()

    if ':' in ra_str or any(c in ra_str for c in 'hms'):
        ra = Angle(ra_str, unit=u.hourangle)
    else:
        ra = Angle(float(ra_str), unit=u.deg)
    
    if any(c in dec_str for c in 'dms') or ':' in dec_str:
        dec = Angle(dec_str, unit=u.deg)
    else:
        dec = Angle(float(dec_str), unit=u.deg)

    return ra.deg, dec.deg

def get_url(*args, **kwargs):
    # Connect and read timeouts
    kwargs["timeout"] = (6.05, 20)
    try:
        return requests.get(*args, **kwargs)
    except requests.exceptions.RequestException:
        return None

        
##### to query star catalogs

def query_stars_gaia(ra,dec,radius = 3):
    '''
    funtion to perform a cone search of stars from Gaia. Queality check on the photometry embedded in the query.
    input
        ra: RA in deg
        dec: Dec in deg
        radius: search radius in arcmin
    returns
        df: Pandas DataFrame with ra, dec, mag_g, and others (source_id, phot_bp_mean_mag, phot_rp_mean_mag, ruwe)
    '''
    
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    
    query = f"""
    SELECT
        source_id,
        ra,
        dec,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag,
        ruwe
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
    
    results
    df = results.to_pandas()
    df.keys()
    df = df.rename(columns={
        "phot_g_mean_mag": "mag_g",
            })

    return df

def query_stars_ps1(ra,dec,radius = 3):
    
    '''
    funtion to perform a cone search of stars from PS1. Queality check on the photometry embedded in the query.
    input
        ra: RA in deg
        dec: Dec in deg
        radius: search radius in arcmin
    returns
        df: Pandas DataFrame with ra, dec, mag_g, mag_r and others (rKronMag,
qualityFlag)
    '''
    
    coord = SkyCoord(ra*u.deg,dec*u.deg)
    
    tbl = Catalogs.query_region(
        coord,
        radius=radius*u.arcmin,
        catalog="Panstarrs",
        table="stack",
        columns=["raMean", "decMean", "gPSFMag", "rPSFMag","rKronMag","qualityFlag"]
    )
    
    tbl = tbl[
        (tbl["rPSFMag"] < 19) 
        &
        (tbl["rPSFMag"] > 14) 
        &
        (abs(tbl["rPSFMag"] - tbl["rKronMag"]) < 0.05) 
        &
        (tbl["qualityFlag"] < 128)
    ]
    
    tbl.sort("rPSFMag")
    tbl[-1]['rPSFMag']
    df = tbl.to_pandas()
    df = df.rename(columns={
    "raMean": "ra",
    "decMean": "dec",
    "gPSFMag": "mag_g",
    "rPSFMag": "mag_r"
        })
    
    return df
    

import pyvo
import astropy.units as u
from astropy.coordinates import SkyCoord
import pyvo
from astropy.table import Table

def query_stars_ls(ra,dec,radius = 6):
    '''
    funtion to perform a box search of stars from Legacy Survey. Queality check on the photometry embedded in the query.
    input    
        ra: RA in deg
        dec: Dec in deg
        radius: size of the box in arcmin
    returns
        df: Pandas DataFrame with ra, dec, mag_g, mag_r, mag_z
    '''
    
    
    # TAP service URL
    tap_url = "https://datalab.noirlab.edu/tap"
    
    # Create TAP service object
    tap_service = pyvo.dal.TAPService(tap_url)
    
    query = f"""
    SELECT TOP 100
        ra, dec,
        mag_g, mag_r, mag_z
    FROM ls_dr10.tractor
    WHERE
        type = 'PSF' 
        AND ra BETWEEN {ra-radius/2/60} AND {ra+radius/2/60}
        AND dec BETWEEN {dec-radius/2/60} AND {dec+radius/2/60}
        AND mag_r < 18
    """
    
    job = tap_service.run_async(query, language="ADQL")
    results = job.to_table()
    
    df = results.to_pandas()
    
    
    return df





