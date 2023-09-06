# Imports
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
from metpy.io import parse_metar_file
from metpy.plots import colortables
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
import matplotlib.pyplot as plt
import numpy as np
import metpy.plots as mpplots
from metpy.units import units
import metpy.calc as mpcalc
import pandas as pd
from io import StringIO
from metpy.units import pandas_dataframe_to_unit_arrays

# Gets the date and time for the analysis

dt = datetime.utcnow()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute
if minute <50:
    dt1 = datetime.utcnow() 
    yr = dt1.year
    mon = dt1.month
    dy = dt1.day
    hr = dt1.hour
    date1 = datetime(yr, mon, dy, hr) - timedelta(hours=2)
    date2 = date1 - timedelta(hours=24)
if minute >=50:
    dt1 = datetime.utcnow() - timedelta(minutes=minute)
    yr = dt1.year
    mon = dt1.month
    dy = dt1.day
    hr = dt1.hour
    date1 = datetime(yr, mon, dy, hr) - timedelta(hours=1)
    date2 = date1 - timedelta(hours=24)

date = datetime(year, month, day, hour, minute)

# Data Access for current time
rtma_cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RTMA/CONUS_2p5km/RTMA_CONUS_2p5km_'+date1.strftime('%Y%m%d_%H00')+'.grib2/catalog.xml')
rtma_data = rtma_cat.datasets['RTMA_CONUS_2p5km_'+date1.strftime('%Y%m%d_%H00')+'.grib2'].remote_access(use_xarray=True)

# Data Access for 24 hours ago
rtma_cat_24 = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RTMA/CONUS_2p5km/RTMA_CONUS_2p5km_'+date2.strftime('%Y%m%d_%H00')+'.grib2/catalog.xml')
rtma_data_24 = rtma_cat_24.datasets['RTMA_CONUS_2p5km_'+date2.strftime('%Y%m%d_%H00')+'.grib2'].remote_access(use_xarray=True)

# Parses data
rtma_data = rtma_data.metpy.parse_cf()
rtma_data_24 = rtma_data_24.metpy.parse_cf()

# Dewpoint data arrays for current time and 24 hours ago
rtma_dwpt = rtma_data['Dewpoint_temperature_error_height_above_ground'].squeeze()

rtma_dwpt_24 = rtma_data_24['Dewpoint_temperature_error_height_above_ground'].squeeze()

# Converts Dewpoint to F
def DWPT_data(rtma_dwpt, rtma_dwpt_24):
    degF = rtma_dwpt * 0.5556
    degF_24 = rtma_dwpt_24 * 0.5556
    
    logic = np.isnan(rtma_dwpt)
    
    if logic.any() == True:       
        return pd.DataFrame()
    if logic.any() == False:
        return degF, degF_24

error, error_24 = DWPT_data(rtma_dwpt, rtma_dwpt_24)

if len(error) != 0:
    plot_proj = error.metpy.cartopy_crs
else:
    pass

# Creates figure
fig = plt.figure(figsize=(14,8))
plt.title("RTMA Dewpoint Temperature Error (\N{DEGREE SIGN}F)\nImage Created: " + dt.strftime('%m/%d/%Y %H:%MZ'), fontsize=20, fontweight='bold')
plt.axis('off')
if len(error) != 0:
    ax = fig.add_subplot(1, 2, 1, projection=plot_proj)
    ax.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(USCOUNTIES, linewidth=0.75)
    ax.set_title("Valid: " + date1.strftime('%m/%d/%Y %HZ'), size=12, fontweight='bold')
    
    # Plots TEMP
    
    cs = ax.contourf(error.metpy.x, error.y, error, 
                     transform=error.metpy.cartopy_crs, levels=np.arange(0, 3.5, 0.5), cmap='viridis')
    cbar_TEMP = fig.colorbar(cs, shrink=0.8)
    cbar_TEMP.set_label(label="Dewpoint Temperature Analysis Error (\N{DEGREE SIGN}F)", size=12, fontweight='bold')

    ax1 = fig.add_subplot(1, 2, 2, projection=plot_proj)
    ax1.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax1.add_feature(cfeature.STATES, linewidth=0.5)
    ax1.add_feature(USCOUNTIES, linewidth=0.75)
    ax1.set_title("Valid: " + date2.strftime('%m/%d/%Y %HZ'), size=12, fontweight='bold')
    # Plots TEMP
    
    cs1 = ax1.contourf(error_24.metpy.x, error_24.y, error_24, 
                     transform=error.metpy.cartopy_crs, levels=np.arange(0, 3.5, 0.5), cmap='viridis')
    cbar_TEMP_24 = fig.colorbar(cs1, shrink=0.8)
    cbar_TEMP_24.set_label(label="Dewpoint Temperature Analysis Error (\N{DEGREE SIGN}F)", size=12, fontweight='bold')
    plt.text(1.25, -0.1, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=14, fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)
    
else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=14, fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   

    
# Saves figure
plt.savefig("Dewpoint Temperature Error")
