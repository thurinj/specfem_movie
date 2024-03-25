#!/usr/bin/env python

"""
Author: Julien Thurin
Date: 25 March 2024
"""

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
import scipy.interpolate
import matplotlib.animation as anim
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.tri as mtri
import matplotlib.collections as mpc
import obspy
from obspy.imaging.beachball import beach
import pyproj

# Configuration Section
# Define all your configurations here. For example:
output_dir = 'path/to/output/'
input_dir = 'path/to/frames' # Where your gmt*.xyz files are stored 
station_file_path = 'path/to/STATIONS' 
topo_path = None # Topo file used in specfem simulation
topo_path = None # Topo file used in specfem simulation
filename = 'vertical_disp'
# Specfem parameters
dt = 0.012
NTSTEP_BETWEEN_FRAMES = 100 
topo_nx, topo_ny = 1024, 512 # If you don't have a topography file and thus no topography dimensions, use NEX_XI and NEX_ETA instead 
easting_min, easting_max = -200000.0, 1200000.0 # These are LONGITUDE_MIN and LONGITUDE_MAX in your Mesh_Par_file
northing_min, northing_max = -100000.0, 600000.0 # These are LATITUDE_MIN and LATITUDE_MAX in your Mesh_Par_file

# Display options
vmax = 1 # Maximum displacement value
vmin = -vmax # Minimum displacement value - So colorbar is centered on 0.
fast_render = True # Will plot only 1 frame out of 10

# OPTIONAL #
# If plotting geographical data, set the paths to the shapefiles and define the UTM zone
coastline_path = None # Set to None if not available
fault_path = None # Set to None if not available
utmzone = None # UTM zone for the projection of shapefiles
hemisphere = None # UTM zone for the projection of shapefiles
rotation_angle = None # Rotation angle for the plot
rotation_origin = None # Origin for the rotation

# global colormap definition
# Custom semi-transparent color map. Change alpha hanning to tweak.
cmap = plt.cm.RdBu
alpha = np.hanning(256)
alpha = np.roll(alpha, int(256/2))
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = alpha
my_cmap = ListedColormap(my_cmap)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

class StationsHelper:
    """
    Represents a station in a seismic network.

    Attributes:
        station_name (str): The name of the station.
        network (str): The network to which the station belongs.
        latitude (str): The latitude of the station.
        longitude (str): The longitude of the station.
    """

    def __init__(self, text_in):
        self.text_split = text_in.split()
        self.station_name = self.text_split[0]
        self.network = self.text_split[1]
        self.latitude = self.text_split[2]
        self.longitude = self.text_split[3]

    @staticmethod
    def split_stations(processed_inputfile_sta):
        """
        Splits the processed input file into a list of station descriptors.

        Args:
            processed_inputfile_sta (list): The processed input file as a list of lines.

        Returns:
            list: A list of station descriptors.
        """
        station_list = []
        for stations_descriptor in processed_inputfile_sta:
            if not stations_descriptor == '':
                station_list.append(stations_descriptor)
        return station_list

    @classmethod
    def load_stations(cls, filepath):
        """
        Loads stations from a file.

        Args:
            filepath (str): The path to the file containing station information.

        Returns:
            list: A list of STATIONS objects representing the stations.
        """
        STATIONS_FILE = open(filepath).read()
        processed_STATIONS_FILE = [line for line in STATIONS_FILE.split('\n')]
        sta_slit = cls.split_stations(processed_STATIONS_FILE)
        stations = [cls(sta) for sta in sta_slit]
        return stations

def get_specfem_gmt_list(dirpath):
    """
    Load and process SPECfem GMT data from a given path.
    """
    gmt_filesname = []
    dirfiles = os.listdir(dirpath)
    for filename in dirfiles:
        if filename.endswith('xyz'):
            gmt_filesname.append(os.path.join(dirpath, filename))
    gmt_filesname.sort()
    return gmt_filesname

def setup_figure(topo_path, topo_nx, topo_ny, extent, coastline_path=None, fault_path=None, projection=None, crs=None):
    """
    Setup the figure for plotting.
    """
    fig = plt.figure(figsize=[10,5])
    ax = fig.add_subplot(111)
    pos1 = ax.get_position()
    pos2 = [pos1.x0 - 0.05, pos1.y0 + 0,  pos1.width / 1, pos1.height / 1]
    ax.set_position(pos2)
    ax.set_aspect('equal')
    easting_min, easting_max = extent[0], extent[1]
    northing_min, northing_max = extent[2], extent[3]
    ax.set_xlim(easting_min, easting_max)
    ax.set_ylim(northing_min, northing_max)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

    # Load and plot topography
    # Define topo background
    if topo_path and topo_nx and topo_ny:
        raw_topo_file = np.loadtxt(topo_path)
        topo = np.flipud(raw_topo_file.reshape(topo_ny, topo_nx))
        lightsource = LightSource(azdeg=315, altdeg=45)
    elif topo_nx and topo_ny:
        topo = np.zeros((topo_ny, topo_nx))
        lightsource = LightSource(azdeg=315, altdeg=45)

    # Plot topography
    im1 = ax.imshow(lightsource.hillshade(topo, vert_exag=0.1, dx=1000, dy=1000), cmap="gray", extent=extent, vmin=0, vmax=1)
    im2 = ax.imshow(np.zeros_like(topo), cmap=my_cmap, extent=extent, vmin=vmin, vmax=vmax)

    # Plot coastline if available   
    if coastline_path and projection and crs:
        print("Plotting coastline ...")
        coastline = gpd.read_file(coastline_path)
        coastline = coastline.to_crs(crs)
        coastline.plot(ax=ax, color='black', linewidth=1)

    # Plot fault lines if available
    if fault_path and projection and crs:
        fault = gpd.read_file(fault_path)
        fault.to_crs(crs)
        fault.plot(ax=ax, color='red', linewidth=1)

    # Plot colorbar
    cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    posn = ax.get_position()
    cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0,
                            0.02, posn.height])
    cbar = plt.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Displacement (m)')

    return fig, ax, my_cmap

# Further modularization and function definitions as needed.
class AnimationHandler:
    def __init__(self, fig, ax, gmt_names, stations, plot_stations=True):
        self.fig = fig
        self.ax = ax
        self.gmt_names = gmt_names
        self.stations = stations
        self.plot_stations = plot_stations
        self.im2 = None
        self.srcs = None

        if fast_render:
            self.gmt_names = self.gmt_names[::10]

    def create_animation(self):
        disp = np.loadtxt(self.gmt_names[0])
        triangles = mtri.Triangulation(disp[:, 0], disp[:, 1])

        global extent

        def update(frame):
            if self.im2:
                for c in self.im2.collections:
                    c.remove()

            print("Processing frame", frame, "of", len(self.gmt_names))
            src = np.loadtxt(self.gmt_names[frame], usecols=2)
            # get the timestep from filename
            filename = os.path.basename(self.gmt_names[frame])
            t_step = int(filename.split(sep="_")[2].split(sep=".")[0]) * NTSTEP_BETWEEN_FRAMES * dt
            self.ax.set_title("time = {:.2f} s".format(t_step))

            if self.plot_stations:
                stations_lat = [float(sta.latitude) for sta in self.stations]
                stations_lon = [float(sta.longitude) for sta in self.stations]
                self.ax.scatter(stations_lon, stations_lat, c="k", marker='v')

            self.im2 = self.ax.tricontourf(triangles,src,512, vmin=vmin, vmax=vmax, cmap=my_cmap, animated=False)

        animation = anim.FuncAnimation(self.fig, update, frames=len(self.gmt_names), repeat=True)
        return animation

def _init_utm_projection(utmzone, hemisphere):
    """
    Initialize the UTM projection based on the zone and hemisphere.
    """
    if hemisphere == 'N':
        utmzone = int(utmzone)
    else:
        utmzone = -int(utmzone)
    myProj = pyproj.Proj(proj='utm', zone=utmzone, ellps='WGS84')
    myCRS = f"+proj=utm +zone={utmzone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    return myProj, myCRS

def main():
    # Initialize projection
    if coastline_path or fault_path:
        myProj, myCRS = _init_utm_projection(utmzone, hemisphere)
    else:
        myProj = None
        myCRS = None
    
    # Load and process data
    stations = StationsHelper.load_stations(station_file_path)
    gmt_names = get_specfem_gmt_list(input_dir)
    extent = (easting_min, easting_max, northing_min, northing_max)
    
    # Setup figure for plotting
    fig, ax, my_cmap = setup_figure(topo_path, topo_nx, topo_ny, extent, coastline_path, fault_path, myProj, myCRS)


    # Create AnimationHandler instance
    anim_handler = AnimationHandler(fig, ax, gmt_names, stations)
    
    # Create animation
    animation = anim_handler.create_animation()
    
    # Save animation
    animation.save(os.path.join(output_dir, f'{filename}.mp4'), fps=24, extra_args=['-vcodec', 'libx264'])

if __name__ == "__main__":
    main()
