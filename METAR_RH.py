# Imports
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
from metpy.io import parse_metar_file
from metpy.plots import colortables
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.plots import USCOUNTIES
import numpy as np
import metpy.plots as mpplots
from metpy.units import units
import metpy.calc as mpcalc
import pandas as pd
from io import StringIO
from metpy.units import pandas_dataframe_to_unit_arrays
from matplotlib.patheffects import withStroke
from metpy.cbook import get_test_data

# Sets date and time
dt = datetime.utcnow()
dt1 = datetime.utcnow()
year = dt.year
month = dt.month
day = dt.day
hour = dt.hour
minute = dt.minute
date = datetime(year, month, day, hour, minute)

if minute <30:
    dt1 = datetime.utcnow()
if minute >=30:
    dt1 = datetime.utcnow() - timedelta(minutes=minute)

# Data access
rtma_cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RTMA/CONUS_2p5km/catalog.xml')
rtma_data = rtma_cat.datasets['Latest Collection for Real Time Mesoscale Analysis 2.5 km'].remote_access(use_xarray=True)
rtma_data = rtma_data.metpy.parse_cf()

# Sets up T and Td dataframes
rtma_temp = rtma_data['Temperature_Analysis_height_above_ground'].metpy.sel(time=dt, method='nearest').squeeze()
rtma_dwpt = rtma_data['Dewpoint_temperature_Analysis_height_above_ground'].metpy.sel(time=dt, method='nearest').squeeze()

# Uses MetPy to create RH dataframe from the T and Td dataframes
rtma_rh = mpcalc.relative_humidity_from_dewpoint(rtma_temp, rtma_dwpt)

# Declares RH for our plot projection
plot_proj = rtma_rh.metpy.cartopy_crs

# Gets airport information
airports_df = pd.read_csv(get_test_data('airport-codes.csv'))
airports_df = airports_df[(airports_df['type'] == 'large_airport') | (airports_df['type'] == 'medium_airport') | (airports_df['type'] == 'small_airport')]

# METAR Data Access
metar_cat = TDSCatalog('https://thredds-test.unidata.ucar.edu/thredds/catalog/noaaport/text/metar/catalog.xml')
metar_file = metar_cat.datasets.filter_time_nearest(dt1).remote_open()
metar_text = StringIO(metar_file.read().decode('latin-1'))
sfc_data = parse_metar_file(metar_text, year=dt1.year, month=dt1.month)
sfc_units = sfc_data.units
sfc_data = sfc_data[sfc_data['station_id'].isin(airports_df['ident'])]
sfc_data = pandas_dataframe_to_unit_arrays(sfc_data, sfc_units)

# Calculates u and v components from wind speed and direction
sfc_data['u'], sfc_data['v'] = mpcalc.wind_components(sfc_data['wind_speed'], sfc_data['wind_direction'])

# Calculates relative humidity from temperature and dewpoint
sfc_data_rh = mpcalc.relative_humidity_from_dewpoint(sfc_data['air_temperature'], sfc_data['dew_point_temperature'])

locs = plot_proj.transform_points(ccrs.PlateCarree(), sfc_data['longitude'].m, sfc_data['latitude'].m)

sfc_data_mask = mpcalc.reduce_point_density(locs[:, :2], 50000)

# Creates figure
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection=plot_proj)
ax.set_extent((-121, -114, 31, 39), crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(USCOUNTIES, linewidth=0.75)

# Plots RH
cs = ax.contourf(rtma_rh.metpy.x, rtma_rh.metpy.y, rtma_rh *100, 
           transform=rtma_rh.metpy.cartopy_crs,
           levels=np.arange(0, 100, 5), cmap='YlGnBu', alpha=0.5)

# Plots METAR
stn = mpplots.StationPlot(ax, sfc_data['longitude'][sfc_data_mask].m, sfc_data['latitude'][sfc_data_mask].m,
                         transform=ccrs.PlateCarree(), fontsize=11, zorder=10, clip_on=True)

# Temperature
stn.plot_parameter('NW', sfc_data['air_temperature'].to('degF')[sfc_data_mask], color='red',
                  path_effects=[withStroke(linewidth=1, foreground='black')])

# Dewpoint
stn.plot_parameter('SW', sfc_data['dew_point_temperature'].to('degF')[sfc_data_mask], color='blue',
                  path_effects=[withStroke(linewidth=1, foreground='black')])

# Sky cover
stn.plot_symbol('C', sfc_data['cloud_coverage'][sfc_data_mask], mpplots.sky_cover)

# Relative Humidity
stn.plot_parameter('E', sfc_data_rh.to('percent')[sfc_data_mask], color='green',
                    path_effects=[withStroke(linewidth=1, foreground='black')])

# Wind barbs
stn.plot_barb(sfc_data['u'][sfc_data_mask], sfc_data['v'][sfc_data_mask])

plt.title("Real Time Mesoscale Analysis(2.5km) Relative Humidity + METAR\nValid: " + dt1.strftime('%m/%d/%Y %HZ') + " | Image Created: " + date.strftime('%m/%d/%Y %H:%MZ'), fontweight='bold')
ax.text(0.5, -0.045, "Developed by Eric Drewitz - Powered by MetPy\nData Source: thredds.ucar.edu", fontweight='bold', horizontalalignment='center',
       verticalalignment='bottom', transform=ax.transAxes)
cbar_RH = fig.colorbar(cs)
cbar_RH.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')

# Saves figure
plt.savefig("METAR and RH")
