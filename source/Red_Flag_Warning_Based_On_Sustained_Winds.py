# Imports
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
from metpy.io import parse_metar_file
from metpy.plots import colortables
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
rtma_speed = rtma_data['Wind_speed_Analysis_height_above_ground'].metpy.sel(time=dt, method='nearest').squeeze()
rtma_speed_mph = rtma_speed * 2.23694

# Calculates RH from the T and Td dataframes using MetPy
rtma_rh = mpcalc.relative_humidity_from_dewpoint(rtma_temp, rtma_dwpt)


# Makes our plot projection to use our RH values
plot_proj = rtma_rh.metpy.cartopy_crs

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


locs = plot_proj.transform_points(ccrs.PlateCarree(), sfc_data['longitude'].m, sfc_data['latitude'].m)

# Creates mask for plotting METAR obs
sfc_data_mask = mpcalc.reduce_point_density(locs[:, :2], 70000)

# Creates Figure
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection=plot_proj)
ax.set_extent((-122, -114, 31, 39), crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(USCOUNTIES, linewidth=0.5)

# Plots RH
cs = ax.contourf(rtma_rh.x, rtma_rh.metpy.y, rtma_rh *100, 
           transform=rtma_rh.metpy.cartopy_crs,
           levels=np.arange(0, 16, 1), cmap='YlOrBr', alpha=0.3, zorder=2)

cs1 = ax.contourf(rtma_speed_mph.x, rtma_speed_mph.metpy.y, rtma_speed_mph,
                  transform=rtma_speed_mph.metpy.cartopy_crs,
                  levels=np.arange(25, 75, 5), cmap='winter', alpha=0.3, zorder=1)

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

cbar_RH = fig.colorbar(cs, location='right', shrink=0.5)
cbar_RH.set_label(label="Relative Humidity (%)", size=12, fontweight='bold')

cbar_gust = fig.colorbar(cs1, location='left', shrink=0.5)
cbar_gust.set_label(label="Wind Speed (mph)", size=12, fontweight='bold')
# Saves figure
plt.savefig(f"Weather Data/RTMA Red Flag Criteria Based on Wind Speed")
