#====================================================================================================
# THIS IS THE MASTER FILE FOR THE 2.5KM X 2.5KM RTMA GRAPHICS
#
# THESE GRAPHICS INCLUDE:
#    1) METAR DATA AND LATEST RTMA RH ANALYSIS
#    2) METAR DATA AND LATEST RTMA RH ANALYSIS FILTERED TO ONLY DISPLAY RH <= 15%
#    3) METAR DATA AND RTMA RFW CONDITIONS BASED ON WIND SPEED
#    4) METAR DATA AND RTMA RFW CONDITIONS BASED ON WIND GUSTS
#    5) LOCALIZED SECTORS (NORTH, CENTRAL AND SOUTH) RTMA RFW CONDITIONS BASED ON WIND SPEED
#    6) LOCALIZED SECTORS (NORTH, CENTRAL AND SOUTH) RTMA RFW CONDITIONS BASED ON WIND GUSTS
#    7) 24 HOUR RH DIFFERENCE USING THE RTMA RH DATA
#    8) 24 HOUR TEMPERATURE DIFFERENCE USING THE RTMA TEMPERATURE DATA
#
#                            |----------------------------------------|
#                            |             DEVELOPED BY               | 
#                            |        (C) ERIC J. DREWITZ             |
#                            |             METEOROLOGIST              |
#                            |               USDA/USFS                |
#                            |----------------------------------------|
#
#=====================================================================================================

# Imports
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
from metpy.io import parse_metar_file
from metpy.plots import colortables
import matplotlib.gridspec as gridspec
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
from matplotlib.patheffects import withStroke
from metpy.cbook import get_test_data

# Dates and times
dt = datetime.utcnow()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute

# Times for METAR reports
if minute <30:
    dt1 = datetime.utcnow() 
if minute >=30:
    dt1 = datetime.utcnow() - timedelta(minutes=minute)
date = datetime(year, month, day, hour, minute)

# Times used for change plots
if minute <50:
    dt2 = datetime.utcnow() 
    yr = dt2.year
    mon = dt2.month
    dy = dt2.day
    hr = dt2.hour
    date1 = datetime(yr, mon, dy, hr) - timedelta(hours=2)
    date2 = date1 - timedelta(hours=24)
if minute >=50:
    dt2 = datetime.utcnow() - timedelta(minutes=minute)
    yr = dt2.year
    mon = dt2.month
    dy = dt2.day
    hr = dt2.hour
    date1 = datetime(yr, mon, dy, hr) - timedelta(hours=1)
    date2 = date1 - timedelta(hours=24)

#-------------------------------------------------------------------------------------------------------
# Data ingest 
#------------------------------------------------------------------------------------------------------

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

# Temperature error data arrays for current time and 24 hours ago
rtma_temp_error = rtma_data['Temperature_error_height_above_ground'].squeeze()
rtma_temp_error_24 = rtma_data_24['Temperature_error_height_above_ground'].squeeze()

# Dewpoint data arrays for current time and 24 hours ago
rtma_dwpt_error = rtma_data['Dewpoint_temperature_error_height_above_ground'].squeeze()
rtma_dwpt_24_error = rtma_data_24['Dewpoint_temperature_error_height_above_ground'].squeeze()

# Wind and wind gust data arrays
rtma_gust = rtma_data['Wind_speed_gust_Analysis_height_above_ground'].squeeze()
rtma_wind = rtma_data['Wind_speed_Analysis_height_above_ground'].squeeze()
rtma_wind_mph = rtma_wind * 2.23694
rtma_gust_mph = rtma_gust * 2.23694

# Function that calculates RH from temperature and dewpoint and then finds the 24 hour RH difference
def rh_data(rtma_temp, rtma_dwpt, rtma_temp_24, rtma_dwpt_24):
    rtma_rh = mpcalc.relative_humidity_from_dewpoint(rtma_temp, rtma_dwpt)
    rtma_rh.to_numpy()
    rtma_rh_24 = mpcalc.relative_humidity_from_dewpoint(rtma_temp_24, rtma_dwpt_24)
    rtma_rh_diff = rtma_rh - rtma_rh_24
    logic = np.isnan(rtma_rh)

    # Returns a blank dataframe if there is no data or returns the data arrays if there is data
    if logic.any() == True:       
        return pd.DataFrame()
    if logic.any() == False:
        return rtma_rh_diff, rtma_rh

rtma_rh_diff, rtma_rh = rh_data(rtma_temp, rtma_dwpt, rtma_temp_24, rtma_dwpt_24)

# Finds temperature difference
def TEMP_data(rtma_temp, rtma_temp_24):
    degC = rtma_temp - 273.15
    frac = 9/5
    degF = (degC * frac) + 32
    degC_24 = rtma_temp_24 - 273.15
    degF_24 = (degC_24 * frac) + 32
    diff = degF - degF_24
    logic = np.isnan(rtma_temp)
    
    if logic.any() == True:       
        return pd.DataFrame()
    if logic.any() == False:
        return diff

rtma_TEMP_diff = TEMP_data(rtma_temp, rtma_temp_24)

# Converts Dewpoint to F
def DWPT_ERROR_data(rtma_dwpt_error, rtma_dwpt_24_error):
    degF = rtma_dwpt_error * 0.5556
    degF_24 = rtma_dwpt_24_error * 0.5556
    
    logic = np.isnan(rtma_dwpt_error)
    
    if logic.any() == True:       
        return pd.DataFrame()
    if logic.any() == False:
        return degF, degF_24

dwpt_error, dwpt_error_24 = DWPT_ERROR_data(rtma_dwpt_error, rtma_dwpt_24_error)

# Converts Temperature to F
def TEMP_ERROR_data(rtma_temp_error, rtma_temp_error_24):
    degF = rtma_temp_error * 0.5556
    degF_24 = rtma_temp_error_24 * 0.5556
    
    logic = np.isnan(rtma_temp_error)
    
    if logic.any() == True:       
        return pd.DataFrame()
    if logic.any() == False:
        return degF, degF_24

temp_error, temp_error_24 = TEMP_ERROR_data(rtma_temp_error, rtma_temp_error_24)

#-------------------------------------------------------------------------------------------------
# Plot projections
#-------------------------------------------------------------------------------------------------

# Makes our plot projection to use our current RH values
if len(rtma_rh) != 0:
    plot_proj_RH = rtma_rh.metpy.cartopy_crs
else:
    pass

# Projection for 24 hour RH change
if len(rtma_rh_diff) != 0:
    plot_proj_RH_Change = rtma_rh_diff.metpy.cartopy_crs
else:
    pass

# Projection for 24 hour temperature change
if len(rtma_TEMP_diff) != 0:
    plot_proj_Temp_Change = rtma_TEMP_diff.metpy.cartopy_crs
else:
    pass

# Projection for temperature error
if len(temp_error) != 0:
    plot_proj_Temp_Error = temp_error.metpy.cartopy_crs
else:
    pass

if len(dwpt_error) != 0:
    plot_proj_dwpt_error = dwpt_error.metpy.cartopy_crs
else:
    pass
    
#---------------------------------------------------------------------------------------------
# METAR SECTION
#---------------------------------------------------------------------------------------------

# Pings server for airport data
airports_df = pd.read_csv(get_test_data('airport-codes.csv'))

# Queries our airport types (airport sizes)
airports_df = airports_df[(airports_df['type'] == 'large_airport') | (airports_df['type'] == 'medium_airport') | (airports_df['type'] == 'small_airport')]

# Accesses the METAR data
metar_cat = TDSCatalog('https://thredds-test.unidata.ucar.edu/thredds/catalog/noaaport/text/metar/catalog.xml')

# Opens METAR file
metar_file = metar_cat.datasets.filter_time_nearest(dt1).remote_open()

# Decodes bytes into strings
metar_text = StringIO(metar_file.read().decode('latin-1'))

# Parses through data
sfc_data = parse_metar_file(metar_text, year=dt.year, month=dt.month)
sfc_units = sfc_data.units

# Creates dataframe
sfc_data = sfc_data[sfc_data['station_id'].isin(airports_df['ident'])]

sfc_data = pandas_dataframe_to_unit_arrays(sfc_data, sfc_units)

sfc_data['u'], sfc_data['v'] = mpcalc.wind_components(sfc_data['wind_speed'], sfc_data['wind_direction'])

sfc_wind = sfc_data['wind_speed'].m

sfc_data_u_kt = sfc_data['u'].to('kts')
sfc_data_v_kt = sfc_data['v'].to('kts')

sfc_data_rh = mpcalc.relative_humidity_from_dewpoint(sfc_data['air_temperature'], sfc_data['dew_point_temperature'])


locs = plot_proj_RH.transform_points(ccrs.PlateCarree(), sfc_data['longitude'].m, sfc_data['latitude'].m)

# Creates mask for plotting METAR obs
sfc_data_mask = mpcalc.reduce_point_density(locs[:, :2], 30000)

#------------------------------------------------------------------------------------------------
# Graphics section
#------------------------------------------------------------------------------------------------

##############################################
# 24 HOUR RH CHANGE PLOT
##############################################

# Creates figure
fig_RH_Change = plt.figure(figsize=(10,10))
if len(rtma_rh_diff) != 0:
    ax = fig_RH_Change.add_subplot(1, 1, 1, projection=plot_proj_RH_Change)
    ax.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(USCOUNTIES, linewidth=0.75)
    
    # Plots RH
    
    cs = ax.contourf(rtma_rh_diff.metpy.x, rtma_rh_diff.y, rtma_rh_diff *100, 
                     transform=rtma_rh_diff.metpy.cartopy_crs,
                     levels=np.arange(-50, 55, 5), cmap='BrBG')
    cbar_RH = fig_RH_Change.colorbar(cs)
    cbar_RH.set_label(label="24-Hour Relative Humidity Change (%)", size=12, fontweight='bold')
    plt.text(0.5, -0.045, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)
else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
plt.title("24-Hour RH Change\nStart: " + date2.strftime('%m/%d/%Y %HZ') + " - End: " + date1.strftime('%m/%d/%Y %HZ') +  "\nImage Created: " + dt.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
plt.axis('off') 

##############################################################################
# 24 HOUR TEMPERATURE CHANGE PLOT
##############################################################################

fig_Temp_Change = plt.figure(figsize=(10,10))
if len(rtma_TEMP_diff) != 0:
    ax = fig_Temp_Change.add_subplot(1, 1, 1, projection=plot_proj_Temp_Change)
    ax.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(USCOUNTIES, linewidth=0.75)
    
    # Plots TEMP
    
    cs = ax.contourf(rtma_TEMP_diff.metpy.x, rtma_TEMP_diff.y, rtma_TEMP_diff, 
                     transform=rtma_TEMP_diff.metpy.cartopy_crs,
                     levels=np.arange(-20, 22, 2), cmap='coolwarm')
    cbar_TEMP = fig_Temp_Change.colorbar(cs)
    cbar_TEMP.set_label(label="24-Hour Temperature Change (\N{DEGREE SIGN}F)", size=12, fontweight='bold')
    plt.text(0.5, -0.045, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)
else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
plt.title("24-Hour Temperature Change (\N{DEGREE SIGN}F)\nStart: " + date2.strftime('%m/%d/%Y %HZ') + " - End: " + date1.strftime('%m/%d/%Y %HZ') +  "\nImage Created: " + dt.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
plt.axis('off') 

##################################################################
# METAR AND CURRENT RTMA RH ANALYSIS
##################################################################

fig_METAR_RH = plt.figure(figsize=(10,10))
if len(rtma_rh) != 0:
    ax = fig_METAR_RH.add_subplot(1, 1, 1, projection=plot_proj_RH)
    ax.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(USCOUNTIES, linewidth=0.75)
    
    # Plots RH
    cs = ax.contourf(rtma_rh.metpy.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 105, 5), cmap='YlGnBu', alpha=0.5)
    
    # Plots METAR
    stn = mpplots.StationPlot(ax, sfc_data['longitude'][sfc_data_mask].m, sfc_data['latitude'][sfc_data_mask].m,
                             transform=ccrs.PlateCarree(), fontsize=11, zorder=10, clip_on=True)
    
    
    stn.plot_parameter('NW', sfc_data['air_temperature'].to('degF')[sfc_data_mask], color='red',
                      path_effects=[withStroke(linewidth=1, foreground='black')])
    
    stn.plot_parameter('SW', sfc_data['dew_point_temperature'].to('degF')[sfc_data_mask], color='blue',
                      path_effects=[withStroke(linewidth=1, foreground='black')])
    
    stn.plot_symbol('C', sfc_data['cloud_coverage'][sfc_data_mask], mpplots.sky_cover)
    
    stn.plot_parameter('E', sfc_data_rh.to('percent')[sfc_data_mask], color='green',
                        path_effects=[withStroke(linewidth=1, foreground='black')])
    
    stn.plot_barb(sfc_data['u'][sfc_data_mask], sfc_data['v'][sfc_data_mask])
    
    plt.title("Real Time Mesoscale Analysis(2.5km) Relative Humidity + METAR\nValid: " + dt1.strftime('%m/%d/%Y %HZ') + " | Image Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
    ax.text(0.5, -0.045, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax.transAxes)
    cbar_RH = fig_METAR_RH.colorbar(cs)
    cbar_RH.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')

else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
    plt.title("Real Time Mesoscale Analysis(2.5km) Relative Humidity + METAR\nValid: " + dt1.strftime('%m/%d/%Y %HZ') + " | Image Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
plt.axis('off') 

######################################################################
# METAR AND RTMA RH (EXCEPTIONALLY LOW RH (RH <= 15%) FILTERED)
######################################################################

fig_METAR_LOW_RH = plt.figure(figsize=(10,10))
if len(rtma_rh) != 0:
    ax = fig_METAR_LOW_RH.add_subplot(1, 1, 1, projection=plot_proj_RH)
    ax.set_extent((-122, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    
    # Plots RH
    cs = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=1)
    
    # Plots METAR
    stn = mpplots.StationPlot(ax, sfc_data['longitude'][sfc_data_mask].m, sfc_data['latitude'][sfc_data_mask].m,
                             transform=ccrs.PlateCarree(), fontsize=11, zorder=10, clip_on=True)
    
    
    stn.plot_parameter('N', sfc_data['wind_speed'].to('kts')[sfc_data_mask], color='red',
                        path_effects=[withStroke(linewidth=1, foreground='black')])
    
    stn.plot_parameter('E', sfc_data_rh.to('percent')[sfc_data_mask], color='green',
                        path_effects=[withStroke(linewidth=1, foreground='black')])
    
    stn.plot_symbol('C', sfc_data['cloud_coverage'][sfc_data_mask], mpplots.sky_cover)
    
    stn.plot_barb(sfc_data_u_kt[sfc_data_mask], sfc_data_v_kt[sfc_data_mask])
    
    plt.title("2.5km Real Time Mesoscale Analysis: Low RH(<=15%)\nMETAR Wind Speed - Red (kts)/RH - Green (%)\nValid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
    ax.text(0.5, -0.045, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax.transAxes)
    
    cbar_RH = fig_METAR_LOW_RH.colorbar(cs)
    cbar_RH.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')

else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
    plt.title("2.5km Real Time Mesoscale Analysis: Low RH(<=15%)\nMETAR Wind Speed - Red (kts)/RH - Green (%)\nValid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
plt.axis('off') 


######################################################################
# METAR AND RTMA RH LOW (RH <= 15%) AND HIGH (RH >= 80%)
######################################################################

fig_LOW_AND_HIGH_RH_AREAS = plt.figure(figsize=(7,7))
if len(rtma_rh) != 0:
    ax = fig_LOW_AND_HIGH_RH_AREAS.add_subplot(1, 1, 1, projection=plot_proj_RH)
    ax.set_extent((-122, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    
    # Plots RH
    cs_low = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=1)

    cs_high = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(80, 101, 1), cmap='Greens', alpha=1)

    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & High RH (RH >= 80%)\nValid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
    ax.text(0.5, -0.077, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax.transAxes)

    cbar_RH_low = fig_LOW_AND_HIGH_RH_AREAS.colorbar(cs_low, location='left', shrink=0.5, pad=0.03)
    cbar_RH_low.set_label(label="Low RH (RH <= 15%)", size=12, fontweight='bold')

    cbar_RH_high = fig_LOW_AND_HIGH_RH_AREAS.colorbar(cs_high, location='right', shrink=0.5, pad=0.03)
    cbar_RH_high.set_label(label="High RH (RH >= 80%)", size=12, fontweight='bold')

else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & High RH (RH >= 80%)\nValid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
plt.axis('off') 


#########################################################################################
# RTMA RFW BASED ON WIND SPEED NO METAR REPORTS
#########################################################################################

fig_RFW_WIND_NO_METAR = plt.figure(figsize=(10,10))
if len(rtma_rh) != 0:
    ax = fig_RFW_WIND_NO_METAR.add_subplot(1, 1, 1, projection=plot_proj_RH)
    ax.set_extent((-122, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
    ax.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)
    
    # Plots RH
    cs = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)
    
    cs1 = ax.contourf(rtma_wind_mph.x, rtma_wind_mph.metpy.y, rtma_wind_mph,
                      transform=rtma_wind_mph.metpy.cartopy_crs,
                      levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)
    

    
    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
    ax.text(0.5, -0.06, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax.transAxes)
    
    cbar_RH = fig_RFW_WIND_NO_METAR.colorbar(cs, location='right', shrink=0.5)
    cbar_RH.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_gust = fig_RFW_WIND_NO_METAR.colorbar(cs1, location='left', shrink=0.5)
    cbar_gust.set_label(label="Wind Speed (mph)", size=12, fontweight='bold')

else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
    plt.axis('off') 



#########################################################################################
# RTMA RFW BASED ON WIND SPEED WITH METARS 
#########################################################################################

fig_RFW_WIND = plt.figure(figsize=(10,10))
if len(rtma_rh) != 0:
    ax = fig_RFW_WIND.add_subplot(1, 1, 1, projection=plot_proj_RH)
    ax.set_extent((-122, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
    ax.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)
    
    # Plots RH
    cs = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)
    
    cs1 = ax.contourf(rtma_wind_mph.x, rtma_wind_mph.metpy.y, rtma_wind_mph,
                      transform=rtma_wind_mph.metpy.cartopy_crs,
                      levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)
    
    # Plots METAR
    stn = mpplots.StationPlot(ax, sfc_data['longitude'][sfc_data_mask].m, sfc_data['latitude'][sfc_data_mask].m,
                             transform=ccrs.PlateCarree(), fontsize=11, zorder=10, clip_on=True)
    
    
    stn.plot_parameter('N', sfc_data['wind_speed'].to('kts')[sfc_data_mask], color='red',
                        path_effects=[withStroke(linewidth=1, foreground='black')])
    
    stn.plot_parameter('E', sfc_data_rh.to('percent')[sfc_data_mask], color='green',
                        path_effects=[withStroke(linewidth=1, foreground='black')])
    
    stn.plot_symbol('C', sfc_data['cloud_coverage'][sfc_data_mask], mpplots.sky_cover)
    
    stn.plot_barb(sfc_data_u_kt[sfc_data_mask], sfc_data_v_kt[sfc_data_mask])
    
    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\n\nMETAR Wind Speed\nRed (kts) & RH - Green (%)\n\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
    ax.text(0.5, -0.06, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax.transAxes)
    
    cbar_RH = fig_RFW_WIND.colorbar(cs, location='right', shrink=0.5)
    cbar_RH.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_gust = fig_RFW_WIND.colorbar(cs1, location='left', shrink=0.5)
    cbar_gust.set_label(label="Wind Speed (mph)", size=12, fontweight='bold')

else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\n\nMETAR Wind Speed\nRed (kts) & RH - Green (%)\n\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
plt.axis('off') 

############################################################################
# RTMA RFW BASED ON WIND GUST NO METAR REPORTS
############################################################################

fig_RFW_GUST_NO_METAR = plt.figure(figsize=(10,10))
if len(rtma_rh) != 0:
    ax = fig_RFW_GUST_NO_METAR.add_subplot(1, 1, 1, projection=plot_proj_RH)
    ax.set_extent((-122, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
    ax.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)
    
    # Plots RH
    cs = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)
    
    cs1 = ax.contourf(rtma_gust_mph.x, rtma_gust_mph.metpy.y, rtma_gust_mph,
                      transform=rtma_gust_mph.metpy.cartopy_crs,
                      levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)
    

    
    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Gusts (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
    ax.text(0.5, -0.06, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax.transAxes)
    
    cbar_RH = fig_RFW_GUST_NO_METAR.colorbar(cs, location='right', shrink=0.5)
    cbar_RH.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_gust = fig_RFW_GUST_NO_METAR.colorbar(cs1, location='left', shrink=0.5)
    cbar_gust.set_label(label="Wind Gust (mph)", size=12, fontweight='bold')

else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Gusts (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
plt.axis('off')



############################################################################
# RTMA RFW BASED ON WIND GUST WITH METARS
############################################################################

fig_RFW_GUST = plt.figure(figsize=(10,10))
if len(rtma_rh) != 0:
    ax = fig_RFW_GUST.add_subplot(1, 1, 1, projection=plot_proj_RH)
    ax.set_extent((-122, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
    ax.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)
    
    # Plots RH
    cs = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)
    
    cs1 = ax.contourf(rtma_gust_mph.x, rtma_gust_mph.metpy.y, rtma_gust_mph,
                      transform=rtma_gust_mph.metpy.cartopy_crs,
                      levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)
    
    # Plots METAR
    stn = mpplots.StationPlot(ax, sfc_data['longitude'][sfc_data_mask].m, sfc_data['latitude'][sfc_data_mask].m,
                             transform=ccrs.PlateCarree(), fontsize=11, zorder=10, clip_on=True)
    
    
    stn.plot_parameter('N', sfc_data['wind_speed'].to('kts')[sfc_data_mask], color='red',
                        path_effects=[withStroke(linewidth=1, foreground='black')])
    
    stn.plot_parameter('E', sfc_data_rh.to('percent')[sfc_data_mask], color='green',
                        path_effects=[withStroke(linewidth=1, foreground='black')])
    
    stn.plot_symbol('C', sfc_data['cloud_coverage'][sfc_data_mask], mpplots.sky_cover)
    
    stn.plot_barb(sfc_data_u_kt[sfc_data_mask], sfc_data_v_kt[sfc_data_mask])
    
    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Gusts (>= 25 mph)\n\nMETAR Wind Speed\nRed (kts) & RH - Green (%)\n\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
    ax.text(0.5, -0.06, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax.transAxes)
    
    cbar_RH = fig_RFW_GUST.colorbar(cs, location='right', shrink=0.5)
    cbar_RH.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_gust = fig_RFW_GUST.colorbar(cs1, location='left', shrink=0.5)
    cbar_gust.set_label(label="Wind Gust (mph)", size=12, fontweight='bold')

else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   
    plt.title("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Gusts (>= 25 mph)\n\nMETAR Wind Speed\nRed (kts) & RH - Green (%)\n\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
plt.axis('off')

##########################################################################################
# RTMA RFW WINDS LOCALIZED SECTORS
##########################################################################################

if len(rtma_rh) != 0:
    # North
    fig_RFW_WIND_NORTH = plt.figure(figsize=(6,3))
    gs = gridspec.GridSpec(10, 10)
    ax = fig_RFW_WIND_NORTH.add_subplot(gs[0:9, 0:9], projection=plot_proj_RH)
    ax.set_extent((-124.6, -117, 36, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
    ax.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)
    
    # Central
    fig_RFW_WIND_CENTRAL = plt.figure(figsize=(6,3))
    ax1 = fig_RFW_WIND_CENTRAL.add_subplot(gs[0:9, 0:9], projection=plot_proj_RH)
    ax1.set_extent((-122, -115, 34, 37), crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
    ax1.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
    ax1.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)
    
    # South
    fig_RFW_WIND_SOUTH = plt.figure(figsize=(6,3))
    ax2 = fig_RFW_WIND_SOUTH.add_subplot(gs[0:9, 0:9], projection=plot_proj_RH)
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
    cbar_RH_north = fig_RFW_WIND_NORTH.colorbar(cs, location='left', shrink=0.8, pad=0.04)
    cbar_RH_north.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_speed_north = fig_RFW_WIND_NORTH.colorbar(cs1, location='right', shrink=0.8)
    cbar_speed_north.set_label(label="Wind Speed (mph)", size=12, fontweight='bold')
    
    cbar_RH_central = fig_RFW_WIND_CENTRAL.colorbar(cs2, location='left', shrink=0.8, pad=0.04)
    cbar_RH_central.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_speed_central = fig_RFW_WIND_CENTRAL.colorbar(cs3, location='right', shrink=0.8)
    cbar_speed_central.set_label(label="Wind Speed (mph)", size=12, fontweight='bold')
    
    cbar_RH_south = fig_RFW_WIND_SOUTH.colorbar(cs4, location='left', shrink=0.8, pad=0.04)
    cbar_RH_south.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_speed_south = fig_RFW_WIND_SOUTH.colorbar(cs5, location='right', shrink=0.8)
    cbar_speed_south.set_label(label="Wind Speed (mph)", size=12, fontweight='bold')
    
    # Plot titles
    fig_RFW_WIND_NORTH.suptitle("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontsize=8, fontweight='bold')
    fig_RFW_WIND_CENTRAL.suptitle("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontsize=8, fontweight='bold')
    fig_RFW_WIND_SOUTH.suptitle("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Speed (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontsize=8, fontweight='bold')
    
    # Plot Signatures
    ax.text(0.5, -0.15, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=7, fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax.transAxes)
    ax1.text(0.5, -0.15, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=7, fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax1.transAxes)
    ax2.text(0.5, -0.15, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=7, fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax2.transAxes)

else:
    pass

##########################################################################################
# RTMA RFW GUSTS LOCALIZED SECTORS
##########################################################################################

if len(rtma_rh) != 0:
    # North
    fig_RFW_GUST_NORTH = plt.figure(figsize=(6,3))
    gs = gridspec.GridSpec(10, 10)
    ax = fig_RFW_GUST_NORTH.add_subplot(gs[0:9, 0:9], projection=plot_proj_RH)
    ax.set_extent((-124.6, -117, 36, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
    ax.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)
    
    # Central
    fig_RFW_GUST_CENTRAL = plt.figure(figsize=(6,3))
    ax1 = fig_RFW_GUST_CENTRAL.add_subplot(gs[0:9, 0:9], projection=plot_proj_RH)
    ax1.set_extent((-122, -115, 34, 37), crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
    ax1.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
    ax1.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)
    
    # South
    fig_RFW_GUST_SOUTH = plt.figure(figsize=(6,3))
    ax2 = fig_RFW_GUST_SOUTH.add_subplot(gs[0:9, 0:9], projection=plot_proj_RH)
    ax2.set_extent((-121, -114, 32, 35), crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75, zorder=3)
    ax2.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
    ax2.add_feature(USCOUNTIES, linewidth=0.5, zorder=3)
    
    
    # Plots RH
    cs = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)
    
    cs1 = ax.contourf(rtma_gust_mph.x, rtma_gust_mph.metpy.y, rtma_gust_mph,
                      transform=rtma_gust_mph.metpy.cartopy_crs,
                      levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)
    
    cs2 = ax1.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)
    
    cs3 = ax1.contourf(rtma_gust_mph.x, rtma_gust_mph.metpy.y, rtma_gust_mph,
                      transform=rtma_gust_mph.metpy.cartopy_crs,
                      levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)
    
    cs4 = ax2.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
               transform=rtma_rh.metpy.cartopy_crs,
               levels=np.arange(0, 16, 1), cmap='YlOrBr_r', alpha=0.9, zorder=2)
    
    cs5 = ax2.contourf(rtma_gust_mph.x, rtma_gust_mph.metpy.y, rtma_gust_mph,
                      transform=rtma_gust_mph.metpy.cartopy_crs,
                      levels=np.arange(25, 75, 5), cmap='winter', alpha=0.5, zorder=1)
    
    # Colorbars
    cbar_RH_north = fig_RFW_GUST_NORTH.colorbar(cs, location='left', shrink=0.8, pad=0.04)
    cbar_RH_north.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_gust_north = fig_RFW_GUST_NORTH.colorbar(cs1, location='right', shrink=0.8)
    cbar_gust_north.set_label(label="Wind Gust (mph)", size=12, fontweight='bold')
    
    cbar_RH_central = fig_RFW_GUST_CENTRAL.colorbar(cs2, location='left', shrink=0.8, pad=0.04)
    cbar_RH_central.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_gust_central = fig_RFW_GUST_CENTRAL.colorbar(cs3, location='right', shrink=0.8)
    cbar_gust_central.set_label(label="Wind Gust (mph)", size=12, fontweight='bold')
    
    cbar_RH_south = fig_RFW_GUST_SOUTH.colorbar(cs4, location='left', shrink=0.8, pad=0.04)
    cbar_RH_south.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')
    
    cbar_gust_south = fig_RFW_GUST_SOUTH.colorbar(cs5, location='right', shrink=0.8)
    cbar_gust_south.set_label(label="Wind Gust (mph)", size=12, fontweight='bold')
    
    # Plot titles
    fig_RFW_GUST_NORTH.suptitle("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Gusts (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontsize=8, fontweight='bold')
    fig_RFW_GUST_CENTRAL.suptitle("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Gusts (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontsize=8, fontweight='bold')
    fig_RFW_GUST_SOUTH.suptitle("2.5km Real Time Mesoscale Analysis\nLow RH(<=15%) & Wind Gusts (>= 25 mph)\nAnalysis Valid: " + dt1.strftime('%m/%d/%Y %HZ') + "\nImage Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontsize=8, fontweight='bold')
    
    # Plot Signatures
    ax.text(0.5, -0.15, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=7, fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax.transAxes)
    ax1.text(0.5, -0.15, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=7, fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax1.transAxes)
    ax2.text(0.5, -0.15, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=7, fontweight='bold', horizontalalignment='center',
           verticalalignment='bottom', transform=ax2.transAxes)

else:
    pass
######################################################################################################
# RTMA TEMPERATURE ERROR
######################################################################################################

fig_TEMP_ERROR = plt.figure(figsize=(14,8))
plt.title("RTMA Temperature Error (\N{DEGREE SIGN}F)\nImage Created: " + dt.strftime('%m/%d/%Y %H:%MZ'), fontsize=20, fontweight='bold')
plt.axis('off')
if len(temp_error) != 0:
    ax = fig_TEMP_ERROR.add_subplot(1, 2, 1, projection=plot_proj_Temp_Error)
    ax.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(USCOUNTIES, linewidth=0.75)
    ax.set_title("Valid: " + date1.strftime('%m/%d/%Y %HZ'), size=12, fontweight='bold')
    
    # Plots TEMP
    
    cs = ax.contourf(temp_error.metpy.x, temp_error.y, temp_error, 
                     transform=temp_error.metpy.cartopy_crs, levels=np.arange(0, 3.5, 0.5), cmap='viridis')
    cbar_TEMP = fig_TEMP_ERROR.colorbar(cs, shrink=0.8)
    cbar_TEMP.set_label(label="Temperature Analysis Error (\N{DEGREE SIGN}F)", size=12, fontweight='bold')

    ax1 = fig_TEMP_ERROR.add_subplot(1, 2, 2, projection=plot_proj_Temp_Error)
    ax1.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax1.add_feature(cfeature.STATES, linewidth=0.5)
    ax1.add_feature(USCOUNTIES, linewidth=0.75)
    ax1.set_title("Valid: " + date2.strftime('%m/%d/%Y %HZ'), size=12, fontweight='bold')
    # Plots TEMP
    
    cs1 = ax1.contourf(temp_error_24.metpy.x, temp_error_24.y, temp_error_24, 
                     transform=temp_error_24.metpy.cartopy_crs, levels=np.arange(0, 3.5, 0.5), cmap='viridis')
    cbar_TEMP_24 = fig_TEMP_ERROR.colorbar(cs1, shrink=0.8)
    cbar_TEMP_24.set_label(label="Temperature Analysis Error (\N{DEGREE SIGN}F)", size=12, fontweight='bold')
    plt.text(1.25, -0.1, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=14, fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)
    
else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=14, fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   

###############################################################################################
# RTMA DEWPOINT ERROR
###############################################################################################

fig_DWPT_ERROR = plt.figure(figsize=(14,8))
plt.title("RTMA Dewpoint Temperature Error (\N{DEGREE SIGN}F)\nImage Created: " + dt.strftime('%m/%d/%Y %H:%MZ'), fontsize=20, fontweight='bold')
plt.axis('off')
if len(dwpt_error) != 0:
    # Creates figure
    ax = fig_DWPT_ERROR.add_subplot(1, 2, 1, projection=plot_proj_dwpt_error)
    ax.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(USCOUNTIES, linewidth=0.75)
    ax.set_title("Valid: " + date1.strftime('%m/%d/%Y %HZ'), size=12, fontweight='bold')
    
    # Plots TEMP
    
    cs = ax.contourf(dwpt_error.metpy.x, dwpt_error.y, dwpt_error, 
                     transform=dwpt_error.metpy.cartopy_crs, levels=np.arange(0, 3.5, 0.5), cmap='viridis')
    cbar_TEMP = fig_DWPT_ERROR.colorbar(cs, shrink=0.8)
    cbar_TEMP.set_label(label="Dewpoint Temperature Analysis Error (\N{DEGREE SIGN}F)", size=12, fontweight='bold')

    ax1 = fig_DWPT_ERROR.add_subplot(1, 2, 2, projection=plot_proj_dwpt_error)
    ax1.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
    ax1.add_feature(cfeature.STATES, linewidth=0.5)
    ax1.add_feature(USCOUNTIES, linewidth=0.75)
    ax1.set_title("Valid: " + date2.strftime('%m/%d/%Y %HZ'), size=12, fontweight='bold')
    # Plots TEMP
    
    cs1 = ax1.contourf(dwpt_error_24.metpy.x, dwpt_error_24.y, dwpt_error_24, 
                     transform=dwpt_error_24.metpy.cartopy_crs, levels=np.arange(0, 3.5, 0.5), cmap='viridis')
    cbar_TEMP_24 = fig_DWPT_ERROR.colorbar(cs1, shrink=0.8)
    cbar_TEMP_24.set_label(label="Dewpoint Temperature Analysis Error (\N{DEGREE SIGN}F)", size=12, fontweight='bold')
    plt.text(1.25, -0.1, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=14, fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)
    
else:
    plt.text(0.2, 0.5, "No Data for " + date.strftime('%m/%d/%Y %HZ'), fontsize=20, fontweight='bold')
    plt.text(0.5, 0, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontsize=14, fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)   


################################################################################################
# THIS SECTION SAVES EACH IMAGE FILE TO A SPECIFIC FOLDER
################################################################################################

fig_RH_Change.savefig(f"Weather Data/24 Hour RH Change")
fig_Temp_Change.savefig(f"Weather Data/24 Hour Temperature Change")
fig_METAR_RH.savefig(f"Weather Data/METAR_And_RH")
fig_METAR_LOW_RH.savefig(f"Weather Data/Exceptionally Low RH")
fig_RFW_WIND.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Speed")
fig_RFW_GUST.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Gusts")
fig_RFW_WIND_NORTH.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Speed North")
fig_RFW_WIND_CENTRAL.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Speed Central")
fig_RFW_WIND_SOUTH.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Speed South")
fig_RFW_GUST_NORTH.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Gust North")
fig_RFW_GUST_CENTRAL.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Gust Central")
fig_RFW_GUST_SOUTH.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Gust South")
fig_TEMP_ERROR.savefig(f"Weather Data/RTMA Temperature Error")
fig_DWPT_ERROR.savefig(f"Weather Data/Dewpoint Temperature Error")
fig_LOW_AND_HIGH_RH_AREAS.savefig(f"Weather Data/Areas of Low and High RH")
fig_RFW_WIND_NO_METAR.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Speed NO METARs")
fig_RFW_GUST_NO_METAR.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Gusts NO METARs")
