# Imports
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
import matplotlib.gridspec as gridspec
from metpy.plots import colortables
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import metpy.plots as mpplots
from metpy.units import units
import metpy.calc as mpcalc
import pandas as pd
from metpy.units import pandas_dataframe_to_unit_arrays
from metpy.plots import USCOUNTIES

# Dates and times
dt = datetime.utcnow()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute
if minute <30:
    dt1 = datetime.utcnow() 
if minute >=30:
    dt1 = datetime.utcnow() - timedelta(minutes=minute)
date = datetime(year, month, day, hour, minute)

# Data Access
rtma_cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RTMA/CONUS_2p5km/catalog.xml')
rtma_data = rtma_cat.datasets['Latest Collection for Real Time Mesoscale Analysis 2.5 km'].remote_access(use_xarray=True)
rtma_data = rtma_data.metpy.parse_cf()

# Extracting the specific dataframes
rtma_temp = rtma_data['Temperature_Analysis_height_above_ground'].metpy.sel(time=dt, method='nearest').squeeze()
rtma_dwpt = rtma_data['Dewpoint_temperature_Analysis_height_above_ground'].metpy.sel(time=dt, method='nearest').squeeze()
rtma_wind = rtma_data['Wind_speed_Analysis_height_above_ground'].metpy.sel(time=dt, method='nearest').squeeze()
rtma_wind_mph = rtma_wind * 2.23694

# Calculates RH from the T and Td dataframes using MetPy
rtma_rh = mpcalc.relative_humidity_from_dewpoint(rtma_temp, rtma_dwpt)

# Makes our plot projection to use our RH values
plot_proj = rtma_rh.metpy.cartopy_crs

# Creates Figures
# North
fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(10, 10)
ax = fig.add_subplot(gs[0:9, 0:9], projection=plot_proj)
ax.set_extent((-124.6, -117, 36, 39), crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
ax.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)

# Central
fig1 = plt.figure(figsize=(6,3))
ax1 = fig1.add_subplot(gs[0:9, 0:9], projection=plot_proj)
ax1.set_extent((-122, -115, 34, 37), crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
ax1.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
ax1.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)

# South
fig2 = plt.figure(figsize=(6,3))
ax2 = fig2.add_subplot(gs[0:9, 0:9], projection=plot_proj)
ax2.set_extent((-121, -114, 32, 35), crs=ccrs.PlateCarree())
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
ax2.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
ax2.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)


# Plots RH
cs = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
           transform=rtma_rh.metpy.cartopy_crs,
           levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)

cs1 = ax.contourf(rtma_wind_mph.x, rtma_wind_mph.metpy.y, rtma_wind_mph,
                  transform=rtma_wind_mph.metpy.cartopy_crs,
                  levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)

cs2 = ax1.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
           transform=rtma_rh.metpy.cartopy_crs,
           levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)

cs3 = ax1.contourf(rtma_wind_mph.x, rtma_wind_mph.metpy.y, rtma_wind_mph,
                  transform=rtma_wind_mph.metpy.cartopy_crs,
                  levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)

cs4 = ax2.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
           transform=rtma_rh.metpy.cartopy_crs,
           levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)

cs5 = ax2.contourf(rtma_wind_mph.x, rtma_wind_mph.metpy.y, rtma_wind_mph,
                  transform=rtma_wind_mph.metpy.cartopy_crs,
                  levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)

# Colorbars
cbar_RH = fig.colorbar(cs, location='left', shrink=0.8, pad=0.04)
cbar_RH.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')

cbar_gust = fig.colorbar(cs1, location='right', shrink=0.8)
cbar_gust.set_label(label="Wind Speed (mph)", size=12, fontweight='bold')

cbar_RH1 = fig1.colorbar(cs2, location='left', shrink=0.8, pad=0.04)
cbar_RH1.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')

cbar_gust1 = fig1.colorbar(cs3, location='right', shrink=0.8)
cbar_gust1.set_label(label="Wind Speed (mph)", size=12, fontweight='bold')

cbar_RH2 = fig2.colorbar(cs4, location='left', shrink=0.8, pad=0.04)
cbar_RH2.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')

cbar_gust2 = fig2.colorbar(cs5, location='right', shrink=0.8)
cbar_gust2.set_label(label="Wind Speed (mph)", size=12, fontweight='bold')

# Plot titles
fig.suptitle("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontsize=8, fontweight='bold')
fig1.suptitle("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontsize=8, fontweight='bold')
fig2.suptitle("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontsize=8, fontweight='bold')

# Plot Signatures
ax.text(0.5, -0.15, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=7, fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)
ax1.text(0.5, -0.15, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=7, fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax1.transAxes)
ax2.text(0.5, -0.15, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=7, fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax2.transAxes)

# Saves figure
fig.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Speed North")
fig1.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Speed Central")
fig2.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Speed South")
