Astronomical Finder Chart Pipeline (Optimized for SOAR/Goodman)

A highly robust, multithreaded Python pipeline designed to automatically generate professional-grade astronomical finder charts. Built with the needs of observational astronomers in mind, this tool creates ready-to-print PDFs featuring optical and infrared sky maps, customized spectrograph slit overlays, and precise star offsets corrected for proper motion.

✨ Key Strengths & Features
Blazing Fast Multithreading: Downloads FITS images (Optical and IR) and queries stellar catalogs (Gaia, 2MASS, etc.) concurrently, drastically reducing the time it takes to generate a chart.

Smart Catalog Fallbacks: Never get a blank map again. If the primary optical catalog (Pan-STARRS) is outside the declination footprint or returns heavily saturated/masked data (like the Orion Nebula), the script automatically falls back to Legacy Survey, DECaPS, or DSS.

Bulletproof Network Resilience: Implements an exponential backoff decorator for database queries. If Gaia temporarily blocks your IP due to high traffic, the script waits and retries automatically instead of crashing.

Proper Motion Corrections: Reference star coordinates retrieved from Gaia DR3 (Epoch 2016.0) are automatically calculated and updated to the current observation year using their pmra and pmdec, ensuring pinpoint accuracy for narrow-slit spectroscopy.

Observatory-Optimized Visuals: Features a dark-adapted friendly layout. Includes direct and rotated views, high-contrast intense colors for reference stars, a 1.0" x 4.0' spectrograph slit overlay, compass roses, scale bars, and an easy-to-read 4x3 offset table.

Dynamic Batch Processing: Process dozens of targets effortlessly using a simple text file. Customize Field of View (fov), Position Angle (pa), and image stretch (contrast) on a per-target basis.

Auto-Cloud Backup: Optionally uploads every generated PDF directly to a specified Google Drive folder so you can access them instantly on the telescope's control computer.

🛠️ Installation & Requirements
It is highly recommended to use an Anaconda environment (Python 3.10+):

Bash

conda create -n finder_env python=3.10
conda activate finder_env
pip install numpy matplotlib astropy astroquery pyvo pandas requests google-api-python-client google-auth-httplib2 google-auth-oauthlib reproject
🚀 How to Use
You can use the program in two ways: for a single target via the command line, or for multiple targets using the batch processor.

Option 1: Single Target (finder.py)
Run the script directly from the terminal for a quick chart.

Bash

python finder.py --s-name "M104" --ra "12:39:59.4" --dec "-11:37:23" --pa-deg 90 --imsize 8 --contrast 0.05
Option 2: Batch Processing (run_batch.py) [Recommended]
Create a text file (e.g., targets.txt) with your targets. You can optionally specify the Position Angle (pa), Field of View (fov), and image contrast at the end of any line. The order of these optional parameters does not matter.

Example targets.txt:

Plaintext

# TargetName    RA             DEC          Options
M104            12:39:59.4     -11:37:23    pa=90 fov=8 contrast=0.05
M42_Orion       05:35:17.3     -05:23:28    pa=45 fov=12 contrast=0.15
M57_Ring        18:53:35.1     +33:01:45    fov=2 contrast=0.03
HD_123456       14:22:33.44    -55:44:33.2 

Then, run the batch script:

Bash

python run_batch.py targets.txt
(The script parses this file natively, avoiding slow subprocess calls, making batch execution incredibly fast).

📂 Google Drive Auto-Upload (Optional)
If you want the script to automatically upload your PDFs to a Google Drive folder:

Obtain a drive_credentials.json Service Account file from the Google Cloud Console.

Place it in the same directory as the scripts.

Share your target Google Drive folder with the Service Account email.

Pass the Folder ID (found in the folder's URL) as an argument:

Bash

python run_batch.py targets.txt --drive-folder 1A2b3C4d5E6f7G8h9I0j
🗺️ Understanding the Output PDF
Each generated PDF is divided into four main quadrants and a data table:

I (Top Left): Optical Direct View (North Up, East Left).

II (Top Right): Optical Rotated View (Aligned to your requested Position Angle).

III (Bottom Left): Infrared Direct View (2MASS).

IV (Bottom Right): Infrared Rotated View.

Visual Markers:

Red Target Crosshair: Centers on your provided RA/DEC.

Green Semi-transparent Rectangle: Represents a standard 1.0" x 4.0' spectrograph slit.

Intense Colored Crosshairs (Yellow, Cyan, Lime): Top 3 reference stars.

Right Panel Table: Provides exact offsets in arcseconds (N/S and E/W) to move the telescope from the reference stars to the target.
