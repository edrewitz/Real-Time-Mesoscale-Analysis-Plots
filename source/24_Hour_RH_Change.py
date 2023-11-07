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

# Temperature and Dewpoint Data Arrays for current time
rtma_temp = rtma_data['Temperature_Analysis_height_above_ground'].squeeze()
rtma_dwpt = rtma_data['Dewpoint_temperature_Analysis_height_above_ground'].squeeze()

# Temperature and Dewpoint Data Arrays for 24 hours ago
rtma_temp_24 = rtma_data_24['Temperature_Analysis_height_above_ground'].squeeze()
rtma_dwpt_24 = rtma_data_24['Dewpoint_temperature_Analysis_height_above_ground'].squeeze()

# Function that calculates RH from temperature and dewpoint and then finds the 24 hour RH difference
def rh_data(rtma_temp, rtma_dwpt, rtma_temp_24, rtma_dwpt_24):
    rtma_rh = mpcalc.relative_humidity_from_dewpoint(rtma_temp, rtma_dwpt)
    rtma_rh.to_numpy()
    rtma_rh_24 = mpcalc.relative_humidity_from_dewpoint(rtma_temp_24, rtma_dwpt_24)
    rtma_rh_diff = rtma_rh - rtma_rh_24
    logic = np.isnan(rtma_rh)
    
    if logic.any() == True:       
        return pd.DataFrame()
    if logic.any() == False:
        return rtma_rh_diff

rtma_rh_diff = rh_data(rtma_temp, rtma_dwpt, rtma_temp_24, rtma_dwpt_24)


if len(rtma_rh_diff) != 0:
    plot_proj = rtma_rh_diff.metpy.cartopy_crs
else:
    pass

# Creates figure
fig = plt.figure(figsize=(10,10))
if len(rtma_rh_diff) != 0:
    ax = fig.add_subplot(1, 1, 1, projection=plot_proj)
    ax.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(USCOUNTIES, linewidth=0.75)
    
    # Plots RH
    
    cs = ax.contourf(rtma_rh_diff.metpy.x, rtma_rh_diff.y, rtma_rh_diff *100, 
                     transform=rtma_rh_diff.metpy.cartopy_crs,
                     levels=np.arange(-50, 55, 5), cmap='BrBG')
    cbar_RH = fig.colorbar(cs)
    cbar_RH.set_label(label="24-Hour Relative Humidity Change (%)", size=12, fontweight='bold')
    plt.text(0.5, -0.045, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)
else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
plt.title("24-Hour RH Change\nStart: " + date2.strftime('%m/%d/%Y %HZ') + " - End: " + date1.strftime('%m/%d/%Y %HZ') +  "\nImage Created: " + dt.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
plt.axis('off') 


# Saves figure
plt.savefig("24_Hour_RH_Change")


